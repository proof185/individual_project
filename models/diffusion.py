"""CondMDI-style diffusion models for motion in-betweening.

This is a drop-in replacement for the previous diffusion.py that moves the
in-betweening model closer to the conditioning scheme in:

  Flexible Motion In-betweening with Diffusion Models (CondMDI)

Key changes relative to the previous version:
- uses masked noisy input x_t_tilde = m * c + (1 - m) * x_t inside the model
- concatenates the observation mask to the model input
- predicts x0 (sample-estimation parameterization)
- uses CFG the same way as before
- keeps the same public API so train.py does not need to change

Important note:
- With the current train.py / dataset interface, keyframes are still whole-frame
  constraints rather than arbitrary partial-joint constraints.
- This file supports feature-level masks internally, but the provided training
  pipeline currently broadcasts keyframes over all features at the selected
  frames.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Standard sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class InbetweenDiffusion:
    """Diffusion process for CondMDI-style motion in-betweening."""

    def __init__(self, T: int, device: str = "cuda"):
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
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        B = x0.shape[0]
        shape = [B] + [1] * (x0.ndim - 1)
        s1 = self.sqrt_alpha_bar[t].view(*shape)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(*shape)
        return s1 * x0 + s2 * noise

    def build_observation_mask(
        self,
        x: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a feature-level observation mask from per-frame keyframe indices.

        Output shape: (B, T, F), with 1 where the keyframe is observed.
        """
        B, T, F = x.shape
        obs = torch.zeros(B, T, F, device=x.device, dtype=x.dtype)

        for b in range(B):
            valid = keyframe_mask[b]
            if valid.any():
                idx = keyframe_indices[b][valid].long()
                obs[b, idx] = 1.0

        return obs

    def masked_mix(
        self,
        xt: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute x_t_tilde = m * c + (1 - m) * x_t and return (x_t_tilde, m)."""
        obs_mask = self.build_observation_mask(xt, keyframe_indices, keyframe_mask)
        xt_tilde = xt.clone()

        for b in range(xt.shape[0]):
            valid = keyframe_mask[b]
            if valid.any():
                idx = keyframe_indices[b][valid].long()
                xt_tilde[b, idx] = keyframes[b][valid]

        return xt_tilde, obs_mask

    def _replace_keyframes(
        self,
        x: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hard replace constrained frames with their given values."""
        x_out = x.clone()
        for b in range(x.shape[0]):
            valid = keyframe_mask[b]
            if valid.any():
                idx = keyframe_indices[b][valid].long()
                x_out[b, idx] = keyframes[b][valid]
        return x_out

    def _predict_x0(
        self,
        model,
        xt: torch.Tensor,
        t_batch: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Model predicts x0 from masked noisy input."""
        return model(
            xt,
            t_batch,
            cond,
            mask,
            keyframes,
            keyframe_indices,
            keyframe_mask,
        )

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
        """Single reverse diffusion step.

        Closer to CondMDI:
        - the model is called on the masked input at every denoising step
        - we do not overwrite x_{t-1} after the reverse step except at t=0,
          where we optionally snap to exact keyframes before returning
        """
        B = xt.shape[0]
        t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

        # Paper-style: mask current x_t with observations before prediction.
        xt_tilde, _ = self.masked_mix(xt, keyframes, keyframe_indices, keyframe_mask)
        xt_tilde = xt_tilde * mask.float().unsqueeze(-1)

        if guidance_scale > 0.0 and cond_uncond is not None:
            x0_uncond = self._predict_x0(
                model, xt_tilde, t_batch, cond_uncond, mask,
                keyframes, keyframe_indices, keyframe_mask
            )
            x0_cond = self._predict_x0(
                model, xt_tilde, t_batch, cond, mask,
                keyframes, keyframe_indices, keyframe_mask
            )
            x0_hat = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        else:
            x0_hat = self._predict_x0(
                model, xt_tilde, t_batch, cond, mask,
                keyframes, keyframe_indices, keyframe_mask
            )

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        if t == 0:
            x_prev = x0_hat
            # Exact final adherence helps downstream use.
            x_prev = self._replace_keyframes(x_prev, keyframes, keyframe_indices, keyframe_mask)
        else:
            alpha_bar_prev = self.alpha_bar[t - 1]
            coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            mean = coef1 * x0_hat + coef2 * xt_tilde
            var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(xt)
            x_prev = mean + torch.sqrt(var) * noise

        x_prev = x_prev * mask.float().unsqueeze(-1)
        return x_prev

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
    ) -> torch.Tensor:
        """Full reverse process."""
        x = torch.randn(shape, device=self.device)
        x = x * mask.float().unsqueeze(-1)

        for t in reversed(range(self.T)):
            x = self.p_sample_inbetween(
                model=model,
                xt=x,
                t=t,
                cond=cond,
                mask=mask,
                keyframes=keyframes,
                keyframe_indices=keyframe_indices,
                keyframe_mask=keyframe_mask,
                guidance_scale=guidance_scale,
                cond_uncond=cond_uncond,
            )

        # Ensure exact keyframes in returned sample.
        x = self._replace_keyframes(x, keyframes, keyframe_indices, keyframe_mask)
        x = x * mask.float().unsqueeze(-1)
        return x


class InbetweenTransformer(nn.Module):
    """CondMDI-style explicit conditional diffusion model.

    Public API intentionally matches the previous implementation so that train.py
    can keep calling:

        x0_hat = model(xt, t, cond, mask, keyframes, keyframe_indices, keyframe_mask)

    Internally the model:
    - forms x_t_tilde = m * c + (1 - m) * x_t
    - concatenates the observation mask m to the input features
    - predicts x0
    """

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

        # CondMDI-style input = [masked xt ; observation mask]
        self.frame_in = nn.Linear(feature_dim * 2, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Time / text conditioning as additive bias token
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

        nn.init.normal_(self.pos_emb, std=0.02)

    def _build_observation_mask(
        self,
        xt: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Feature-level observed mask (B, T, F)."""
        B, T, F = xt.shape
        obs = torch.zeros(B, T, F, device=xt.device, dtype=xt.dtype)
        for b in range(B):
            valid = keyframe_mask[b]
            if valid.any():
                idx = keyframe_indices[b][valid].long()
                obs[b, idx] = 1.0
        return obs

    def _build_keyframe_canvas(
        self,
        xt: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dense c tensor with zeros at unobserved locations."""
        canvas = torch.zeros_like(xt)
        for b in range(xt.shape[0]):
            valid = keyframe_mask[b]
            if valid.any():
                idx = keyframe_indices[b][valid].long()
                canvas[b, idx] = keyframes[b][valid]
        return canvas

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
        if F != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {F}."
            )
        if T > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds max_len {self.pos_emb.shape[1]}."
            )

        obs_mask = self._build_observation_mask(xt, keyframe_indices, keyframe_mask)
        keyframe_canvas = self._build_keyframe_canvas(xt, keyframes, keyframe_indices, keyframe_mask)

        # CondMDI masked addition:
        # x_t_tilde = m * c + (1 - m) * x_t
        xt_tilde = obs_mask * keyframe_canvas + (1.0 - obs_mask) * xt
        xt_tilde = xt_tilde * mask.float().unsqueeze(-1)
        obs_mask = obs_mask * mask.float().unsqueeze(-1)

        # Concatenate masked motion and observation mask.
        inp = torch.cat([xt_tilde, obs_mask], dim=-1)

        h = self.frame_in(inp)
        h = h + self.pos_emb[:, :T, :]

        t_emb = timestep_embedding(t, self.d_model)
        cond_token = (self.t_mlp(t_emb) + self.c_mlp(cond)).unsqueeze(1)

        tokens = torch.cat([cond_token, h], dim=1)

        src_key_padding_mask = torch.cat(
            [
                torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                ~mask,
            ],
            dim=1,
        )

        y = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        y = y[:, 1:, :]
        x0_hat = self.out(y)
        x0_hat = x0_hat * mask.float().unsqueeze(-1)
        return x0_hat
