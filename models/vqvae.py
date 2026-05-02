"""VQ-VAE models for motion tokenization.

Architecture:
  Encoder: Conv1d stem → 2x ResBlock → stride-2 downsample → 2x ResBlock
           → stride-2 downsample → 2x ResBlock → projection to codebook_dim
  VQ layer: EMA-updated codebook (1024 entries by default)
  Decoder: projection from codebook_dim → 2x ResBlock → stride-2 upsample
           → 2x ResBlock → stride-2 upsample → 2x ResBlock → output Conv1d
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    """Return a group count compatible with GroupNorm (divisor of channels, <= 32)."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock1d(nn.Module):
    """1-D residual block with GroupNorm and SiLU activations."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        g = _num_groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)

        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )

        indices = d.argmin(dim=1)
        z_q = self.embedding(indices).view(B, T, D)

        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_embeddings).float()
                self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
                dw = z_flat.t() @ encodings
                self.ema_w.mul_(self.decay).add_(dw.t(), alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        commitment_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()  # Straight-through

        indices = indices.view(B, T)
        return z_q, indices, self.commitment_cost * commitment_loss


class MotionVQVAE(nn.Module):
    """VQ-VAE for motion tokenization with residual encoder/decoder.

    The encoder and decoder each use ResBlock1d blocks (GroupNorm + SiLU)
    around stride-2 conv / transposed-conv downsampling layers, giving the
    model much stronger reconstruction capacity than a plain sequential stack.
    """

    def __init__(
        self,
        feature_dim: int,
        codebook_size: int = 1024,
        codebook_dim: int = 512,
        downsample_rate: int = 4,
        commitment_cost: float = 0.25,
        enc_channels: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_dim = codebook_dim
        self.downsample_rate = downsample_rate
        C = enc_channels

        # ----- Encoder -----
        self.enc_in = nn.Conv1d(feature_dim, C, kernel_size=3, padding=1)
        self.enc_res1 = nn.Sequential(ResBlock1d(C, dropout), ResBlock1d(C, dropout))
        self.enc_down1 = nn.Conv1d(C, C, kernel_size=4, stride=2, padding=1)
        self.enc_res2 = nn.Sequential(ResBlock1d(C, dropout), ResBlock1d(C, dropout))
        self.enc_down2 = nn.Conv1d(C, C, kernel_size=4, stride=2, padding=1)
        self.enc_res3 = nn.Sequential(ResBlock1d(C, dropout), ResBlock1d(C, dropout))
        self.enc_out = nn.Conv1d(C, codebook_dim, kernel_size=3, padding=1)

        self.vq = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)

        # ----- Decoder -----
        self.dec_in = nn.Conv1d(codebook_dim, C, kernel_size=3, padding=1)
        self.dec_res1 = nn.Sequential(ResBlock1d(C, dropout), ResBlock1d(C, dropout))
        self.dec_up1 = nn.ConvTranspose1d(C, C, kernel_size=4, stride=2, padding=1)
        self.dec_res2 = nn.Sequential(ResBlock1d(C, dropout), ResBlock1d(C, dropout))
        self.dec_up2 = nn.ConvTranspose1d(C, C, kernel_size=4, stride=2, padding=1)
        self.dec_res3 = nn.Sequential(ResBlock1d(C, dropout), ResBlock1d(C, dropout))
        self.dec_out = nn.Conv1d(C, feature_dim, kernel_size=3, padding=1)

    # ------------------------------------------------------------------
    # Encoder path (returns channel-last tensors for VQ)
    # ------------------------------------------------------------------

    def _encode_conv(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → z: (B, T', codebook_dim) channel-last."""
        h = x.transpose(1, 2)             # (B, F, T)
        h = self.enc_in(h)
        h = self.enc_res1(h)
        h = F.silu(self.enc_down1(h))
        h = self.enc_res2(h)
        h = F.silu(self.enc_down2(h))
        h = self.enc_res3(h)
        h = self.enc_out(h)
        return h.transpose(1, 2)          # (B, T', codebook_dim)

    # ------------------------------------------------------------------
    # Decoder path (accepts channel-last z_q)
    # ------------------------------------------------------------------

    def _decode_conv(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, T', codebook_dim) → x_recon: (B, T, F)."""
        h = z_q.transpose(1, 2)           # (B, codebook_dim, T')
        h = self.dec_in(h)
        h = self.dec_res1(h)
        h = F.silu(self.dec_up1(h))
        h = self.dec_res2(h)
        h = F.silu(self.dec_up2(h))
        h = self.dec_res3(h)
        h = self.dec_out(h)
        return h.transpose(1, 2)          # (B, T, F)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._encode_conv(x)
        z_q, indices, _ = self.vq(z)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self._decode_conv(z_q)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.vq.embedding(indices)
        return self._decode_conv(z_q)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._encode_conv(x)
        z_q, indices, vq_loss = self.vq(z)
        x_recon = self._decode_conv(z_q)
        return x_recon, indices, vq_loss
