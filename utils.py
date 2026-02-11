"""Utility functions for training and evaluation."""

from typing import List

import torch
import torch.nn.functional as F
import clip


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss with masking."""
    m = mask.float().unsqueeze(-1)
    se = (a - b) ** 2
    se = se * m
    denom = m.sum() * se.shape[-1] + 1e-8
    return se.sum() / denom


def vel_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Velocity loss for smooth motion."""
    m = mask[:, 1:] & mask[:, :-1]
    v_gt = x[:, 1:] - x[:, :-1]
    v_pr = x_hat[:, 1:] - x_hat[:, :-1]
    return masked_mse(v_gt, v_pr, m)


def setup_clip_model(device: str = 'cuda'):
    """Load and setup CLIP model for text encoding."""
    clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model


@torch.no_grad()
def encode_text(clip_model, texts: List[str], normalize: bool = True) -> torch.Tensor:
    """Encode text using CLIP."""
    device = next(clip_model.parameters()).device
    tokens = clip.tokenize(texts, truncate=True).to(device)
    emb = clip_model.encode_text(tokens).float()
    if normalize:
        emb = F.normalize(emb, dim=-1)
    return emb


def prepare_gpt_batch(batch, vqvae, gpt, downsample_rate, device):
    """Prepare batch for GPT training."""
    x = batch['motion'].to(device)
    mask = batch['mask'].to(device)
    B, T, _ = x.shape

    pad_len = (downsample_rate - T % downsample_rate) % downsample_rate
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))

    with torch.no_grad():
        indices, _ = vqvae.encode(x)

    T_tokens = indices.shape[1]
    token_lengths = (batch['lengths'].to(device) + downsample_rate - 1) // downsample_rate
    token_mask = torch.arange(T_tokens, device=device).unsqueeze(0) < token_lengths.unsqueeze(1)

    bos = torch.full((B, 1), gpt.bos_token, dtype=torch.long, device=device)
    eos = torch.full((B, 1), gpt.eos_token, dtype=torch.long, device=device)

    input_tokens = torch.cat([bos, indices], dim=1)
    target_tokens = torch.cat([indices, eos], dim=1)
    target_mask = torch.cat([token_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)

    return input_tokens, target_tokens, target_mask
