"""VQ-VAE models for motion tokenization."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """VQ-VAE for motion tokenization."""
    def __init__(
        self,
        feature_dim: int,
        codebook_size: int = 512,
        codebook_dim: int = 512,
        downsample_rate: int = 4,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_dim = codebook_dim
        self.downsample_rate = downsample_rate

        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, codebook_dim, kernel_size=3, padding=1),
        )

        self.vq = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Conv1d(codebook_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, feature_dim, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        z = self.encoder(x)
        z = z.transpose(1, 2)
        z_q, indices, _ = self.vq(z)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z_q = z_q.transpose(1, 2)
        x = self.decoder(z_q)
        x = x.transpose(1, 2)
        return x

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.vq.embedding(indices)
        return self.decode(z_q)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_conv = x.transpose(1, 2)
        z = self.encoder(x_conv)
        z = z.transpose(1, 2)

        z_q, indices, vq_loss = self.vq(z)
        x_recon = self.decode(z_q)

        return x_recon, indices, vq_loss
