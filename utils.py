"""Utility functions for training and evaluation."""

from typing import List

import torch
import torch.nn.functional as F
import clip


def encode_text_with_tokens(clip_model, texts: List[str], normalize: bool = True):
    """Encode text into pooled CLIP embeddings and per-token features."""
    device = next(clip_model.parameters()).device
    tokens = clip.tokenize(texts, truncate=True).to(device)

    x = clip_model.token_embedding(tokens).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)

    if hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
        token_features = x @ clip_model.text_projection
    else:
        token_features = x

    pooled = token_features[torch.arange(token_features.shape[0], device=device), tokens.argmax(dim=-1)]
    token_mask = tokens.ne(0)

    token_features = token_features.float()
    pooled = pooled.float()
    if normalize:
        pooled = F.normalize(pooled, dim=-1)
        token_features = F.normalize(token_features, dim=-1)

    return pooled, token_features, token_mask


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
    """Match velocity only on edges touching keyframes (vectorized)."""
    B, T, _ = x.shape
    edge_len = max(T - 1, 0)
    if edge_len == 0:
        return x.new_tensor(0.0)

    idx = keyframe_indices.long()   # (B, K)
    valid_k = keyframe_mask.bool()  # (B, K)
    mask_long = mask.long()         # (B, T)

    idx_clamp  = idx.clamp(0, T - 1)
    left_idx   = (idx - 1).clamp(0, T - 1)
    right_idx  = (idx + 1).clamp(0, T - 1)

    at_kf    = torch.gather(mask_long, 1, idx_clamp).bool()  # (B, K)
    at_left  = torch.gather(mask_long, 1, left_idx).bool()   # (B, K)
    at_right = torch.gather(mask_long, 1, right_idx).bool()  # (B, K)

    # Left edge (idx-1 → idx): edge position idx-1
    left_valid = valid_k & (idx > 0) & at_kf & at_left
    left_pos   = (idx - 1).clamp(0, edge_len - 1)
    # Right edge (idx → idx+1): edge position idx
    right_valid = valid_k & (idx < T - 1) & at_kf & at_right
    right_pos   = idx.clamp(0, edge_len - 1)

    edge_float = x.new_zeros(B, edge_len)
    edge_float.scatter_add_(1, left_pos,  left_valid.float())
    edge_float.scatter_add_(1, right_pos, right_valid.float())
    edge_mask = edge_float > 0.0

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
    """Match acceleration only in triplets centred on keyframes (vectorized)."""
    B, T, _ = x.shape
    accel_len = max(T - 2, 0)
    if accel_len == 0:
        return x.new_tensor(0.0)

    idx = keyframe_indices.long()   # (B, K)
    valid_k = keyframe_mask.bool()  # (B, K)
    mask_long = mask.long()         # (B, T)

    idx_clamp  = idx.clamp(0, T - 1)
    left_idx   = (idx - 1).clamp(0, T - 1)
    right_idx  = (idx + 1).clamp(0, T - 1)

    at_kf    = torch.gather(mask_long, 1, idx_clamp).bool()  # (B, K)
    at_left  = torch.gather(mask_long, 1, left_idx).bool()   # (B, K)
    at_right = torch.gather(mask_long, 1, right_idx).bool()  # (B, K)

    # Triplet centred at idx: accel tensor index = idx - 1
    triplet_valid = valid_k & (idx > 0) & (idx < T - 1) & at_kf & at_left & at_right
    accel_pos = (idx - 1).clamp(0, accel_len - 1)

    accel_float = x.new_zeros(B, accel_len)
    accel_float.scatter_add_(1, accel_pos, triplet_valid.float())
    accel_mask = accel_float > 0.0

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
    emb, _, _ = encode_text_with_tokens(clip_model, texts, normalize=normalize)
    return emb


@torch.no_grad()
def encode_text_tokens(clip_model, texts: List[str], normalize: bool = True):
    """Encode text into per-token CLIP features plus a valid-token mask."""
    _, token_features, token_mask = encode_text_with_tokens(clip_model, texts, normalize=normalize)
    return token_features, token_mask


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
