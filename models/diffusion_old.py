"""Diffusion models for motion in-betweening."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule for diffusion."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class InbetweenDiffusion:
    """Diffusion process for in-betweening."""
    def __init__(self, T: int, device: str = 'cuda'):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register(betas=betas, alphas=alphas, alpha_bar=alpha_bar)

    def register(self, **tensors):
        for k, v in tensors.items():
            setattr(self, k, v.to(self.device))

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Add noise to x0 according to timestep t."""
        B = x0.shape[0]
        shape = [B] + [1] * (x0.ndim - 1)
        s1 = self.sqrt_alpha_bar[t].view(*shape)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(*shape)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample_inbetween(
        self,
        model,
        xt: torch.Tensor,
        t: int,
        cond: torch.Tensor,
        mask: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
        guidance_scale: float = 0.0,
        cond_uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single denoising step."""
        B = xt.shape[0]
        t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

        # Predict x0
        if guidance_scale > 0 and cond_uncond is not None:
            x0_uncond = model(xt, t_batch, cond_uncond, mask, keyframes, keyframe_indices, keyframe_mask)
            x0_cond = model(xt, t_batch, cond, mask, keyframes, keyframe_indices, keyframe_mask)
            x0_hat = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        else:
            x0_hat = model(xt, t_batch, cond, mask, keyframes, keyframe_indices, keyframe_mask)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        if t == 0:
            x_prev = x0_hat
        else:
            coef1 = torch.sqrt(self.alpha_bar[t - 1]) * beta_t / (1 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1 - self.alpha_bar[t - 1]) / (1 - alpha_bar_t)
            mean = coef1 * x0_hat + coef2 * xt
            var = beta_t * (1 - self.alpha_bar[t - 1]) / (1 - alpha_bar_t)
            noise = torch.randn_like(xt)
            x_prev = mean + torch.sqrt(var) * noise

        # Replace keyframe positions with actual keyframe values
        x_prev = self._replace_keyframes(x_prev, keyframes, keyframe_indices, keyframe_mask)

        x_prev = x_prev * mask.float().unsqueeze(-1)
        return x_prev

    def _replace_keyframes(
        self,
        x: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace keyframe positions with ground truth values."""
        B, T, F = x.shape
        x_out = x.clone()

        for b in range(B):
            valid_mask = keyframe_mask[b]
            valid_indices = keyframe_indices[b][valid_mask]
            valid_keyframes = keyframes[b][valid_mask]
            x_out[b, valid_indices] = valid_keyframes

        return x_out

    @torch.no_grad()
    def sample_inbetween(
        self,
        model,
        shape: Tuple[int, int, int],
        cond: torch.Tensor,
        mask: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
        guidance_scale: float = 2.5,
        cond_uncond: Optional[torch.Tensor] = None,
    ):
        """Complete sampling process."""
        x = torch.randn(shape, device=self.device)

        # Initialize with keyframes
        x = self._replace_keyframes(x, keyframes, keyframe_indices, keyframe_mask)
        x = x * mask.float().unsqueeze(-1)

        for t in reversed(range(self.T)):
            x = self.p_sample_inbetween(
                model, x, t, cond, mask,
                keyframes, keyframe_indices, keyframe_mask,
                guidance_scale=guidance_scale,
                cond_uncond=cond_uncond
            )

        return x


class InbetweenTransformer(nn.Module):
    """Transformer model for diffusion in-betweening."""
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int = 512,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        # Input projection
        self.frame_in = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Keyframe indicator embedding (learnable)
        # 0 = not keyframe, 1 = keyframe
        self.keyframe_emb = nn.Embedding(2, d_model)

        # Keyframe value encoder (separate from noisy input)
        self.keyframe_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Text condition
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Output
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, F = xt.shape

        # Create keyframe indicator for each frame
        keyframe_indicator = torch.zeros(B, T, dtype=torch.long, device=xt.device)
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            keyframe_indicator[b, valid_idx] = 1

        # Encode noisy input
        h = self.frame_in(xt)  # (B, T, d)
        h = h + self.pos_emb[:, :T, :]

        # Add keyframe indicator embedding
        h = h + self.keyframe_emb(keyframe_indicator)

        # Inject keyframe values at their positions
        keyframe_features = self.keyframe_encoder(keyframes)  # (B, K, d)
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            h[b, valid_idx] = h[b, valid_idx] + keyframe_features[b, valid]

        # Time + condition token
        t_emb = timestep_embedding(t, self.d_model)
        zt = self.t_mlp(t_emb) + self.c_mlp(cond)
        zt = zt.unsqueeze(1)  # (B, 1, d)

        # Prepend conditioning token
        tokens = torch.cat([zt, h], dim=1)  # (B, 1+T, d)

        # Padding mask
        src_key_padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
            ~mask
        ], dim=1)

        # Transformer
        y = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        y = y[:, 1:, :]  # Remove condition token

        # Output prediction
        x0_hat = self.out(y)
        x0_hat = x0_hat * mask.float().unsqueeze(-1)

        return x0_hat
