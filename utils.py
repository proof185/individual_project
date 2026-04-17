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


def weighted_masked_mse(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted MSE on valid positions only."""
    w = weights.float() * valid_mask.float()
    w = w.unsqueeze(-1)
    se = (a - b) ** 2
    se = se * w
    denom = w.sum() * se.shape[-1] + 1e-8
    return se.sum() / denom


def vel_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Velocity loss for smooth motion."""
    m = mask[:, 1:] & mask[:, :-1]
    v_gt = x[:, 1:] - x[:, :-1]
    v_pr = x_hat[:, 1:] - x_hat[:, :-1]
    return masked_mse(v_gt, v_pr, m)


def jerk_loss(x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Jerk loss - penalizes sudden changes in acceleration (third derivative)."""
    # Jerk = third derivative (change in acceleration)
    vel = torch.diff(x_hat, dim=1)          # (B, T-1, F)
    accel = torch.diff(vel, dim=1)          # (B, T-2, F)
    jerk = torch.diff(accel, dim=1)         # (B, T-3, F)
    jerk_mask = mask[:, 3:]
    # Penalize non-zero jerk
    return masked_mse(torch.zeros_like(jerk), jerk, jerk_mask)


def boundary_velocity_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor,
    keyframe_indices: torch.Tensor,
    keyframe_mask: torch.Tensor,
) -> torch.Tensor:
    """Match velocity only on edges touching keyframes."""
    B, T, _ = x.shape
    edge_mask = torch.zeros(B, max(T - 1, 0), dtype=torch.bool, device=x.device)

    for b in range(B):
        valid_idx = keyframe_indices[b][keyframe_mask[b]]
        for idx in valid_idx.tolist():
            if idx > 0 and mask[b, idx - 1] and mask[b, idx]:
                edge_mask[b, idx - 1] = True
            if idx < T - 1 and mask[b, idx] and mask[b, idx + 1]:
                edge_mask[b, idx] = True

    v_gt = x[:, 1:] - x[:, :-1]
    v_pr = x_hat[:, 1:] - x_hat[:, :-1]
    return masked_mse(v_gt, v_pr, edge_mask)


def boundary_acceleration_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor,
    keyframe_indices: torch.Tensor,
    keyframe_mask: torch.Tensor,
) -> torch.Tensor:
    """Match acceleration only in triplets centered on keyframes."""
    B, T, _ = x.shape
    accel_mask = torch.zeros(B, max(T - 2, 0), dtype=torch.bool, device=x.device)

    for b in range(B):
        valid_idx = keyframe_indices[b][keyframe_mask[b]]
        for idx in valid_idx.tolist():
            if 0 < idx < T - 1 and mask[b, idx - 1] and mask[b, idx] and mask[b, idx + 1]:
                accel_mask[b, idx - 1] = True

    accel_gt = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    accel_pr = x_hat[:, 2:] - 2 * x_hat[:, 1:-1] + x_hat[:, :-2]
    return masked_mse(accel_gt, accel_pr, accel_mask)


def transition_consistency_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor,
    keyframe_indices: torch.Tensor,
    keyframe_mask: torch.Tensor,
    window: int = 4,
    sigma: float = 1.5,
    velocity_weight: float = 1.0,
    acceleration_weight: float = 0.5,
) -> torch.Tensor:
    """Penalize velocity/acceleration mismatch in a weighted window around keyframes.

    This targets snapping artifacts more directly than a global jerk penalty.
    """
    B, T, _ = x.shape
    if T < 3 or window <= 0:
        return x.new_tensor(0.0)
    if velocity_weight <= 0 and acceleration_weight <= 0:
        return x.new_tensor(0.0)
    if sigma <= 0:
        sigma = 1.0

    edge_len = max(T - 1, 0)
    accel_len = max(T - 2, 0)

    # Vectorized keyframe-local Gaussian weighting to avoid Python loops.
    k_idx = keyframe_indices.to(device=x.device)
    k_valid = keyframe_mask.to(device=x.device).bool()
    if not k_valid.any():
        return x.new_tensor(0.0)
    gaussian_scale = -0.5 / (sigma * sigma)

    if edge_len > 0:
        edge_pos = torch.arange(edge_len, device=x.device).view(1, 1, edge_len)
        k_expanded = k_idx.unsqueeze(-1)
        edge_dist = torch.minimum((edge_pos - k_expanded).abs(), (edge_pos + 1 - k_expanded).abs()).float()
        edge_valid = k_valid.unsqueeze(-1) & (edge_dist <= window)
        edge_per_key = torch.exp(edge_dist.square() * gaussian_scale) * edge_valid.float()
        edge_weights = edge_per_key.max(dim=1).values
    else:
        edge_weights = torch.zeros(B, 0, device=x.device)

    if accel_len > 0:
        accel_center = (torch.arange(accel_len, device=x.device) + 1).view(1, 1, accel_len)
        k_expanded = k_idx.unsqueeze(-1)
        accel_dist = (accel_center - k_expanded).abs().float()
        accel_valid = k_valid.unsqueeze(-1) & (accel_dist <= window)
        accel_per_key = torch.exp(accel_dist.square() * gaussian_scale) * accel_valid.float()
        accel_weights = accel_per_key.max(dim=1).values
    else:
        accel_weights = torch.zeros(B, 0, device=x.device)

    if velocity_weight > 0:
        v_gt = x[:, 1:] - x[:, :-1]
        v_pr = x_hat[:, 1:] - x_hat[:, :-1]
        valid_edges = mask[:, 1:] & mask[:, :-1]
        vel_term = weighted_masked_mse(v_gt, v_pr, edge_weights, valid_edges)
    else:
        vel_term = x.new_tensor(0.0)

    if acceleration_weight > 0:
        accel_gt = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        accel_pr = x_hat[:, 2:] - 2 * x_hat[:, 1:-1] + x_hat[:, :-2]
        valid_accel = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        accel_term = weighted_masked_mse(accel_gt, accel_pr, accel_weights, valid_accel)
    else:
        accel_term = x.new_tensor(0.0)

    return velocity_weight * vel_term + acceleration_weight * accel_term


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
