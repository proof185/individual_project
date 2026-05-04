"""Utility functions for training and evaluation."""

import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import clip

from condmdi_adapter import load_external_condmdi_runtime, looks_like_condmdi_checkpoint


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


def boundary_jerk_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor,
    keyframe_indices: torch.Tensor,
    keyframe_mask: torch.Tensor,
) -> torch.Tensor:
    """Match jerk only on four-frame stencils immediately surrounding keyframes."""
    B, T, _ = x.shape
    jerk_len = max(T - 3, 0)
    if jerk_len == 0:
        return x.new_tensor(0.0)

    idx = keyframe_indices.long()
    valid_k = keyframe_mask.bool()
    mask_long = mask.long()

    idx_m2 = (idx - 2).clamp(0, T - 1)
    idx_m1 = (idx - 1).clamp(0, T - 1)
    idx_clamp = idx.clamp(0, T - 1)
    idx_p1 = (idx + 1).clamp(0, T - 1)
    idx_p2 = (idx + 2).clamp(0, T - 1)

    at_m2 = torch.gather(mask_long, 1, idx_m2).bool()
    at_m1 = torch.gather(mask_long, 1, idx_m1).bool()
    at_kf = torch.gather(mask_long, 1, idx_clamp).bool()
    at_p1 = torch.gather(mask_long, 1, idx_p1).bool()
    at_p2 = torch.gather(mask_long, 1, idx_p2).bool()

    left_valid = valid_k & (idx > 1) & (idx < T - 1) & at_m2 & at_m1 & at_kf & at_p1
    left_pos = (idx - 2).clamp(0, jerk_len - 1)

    right_valid = valid_k & (idx > 0) & (idx < T - 2) & at_m1 & at_kf & at_p1 & at_p2
    right_pos = (idx - 1).clamp(0, jerk_len - 1)

    jerk_float = x.new_zeros(B, jerk_len)
    jerk_float.scatter_add_(1, left_pos, left_valid.float())
    jerk_float.scatter_add_(1, right_pos, right_valid.float())
    jerk_mask = jerk_float > 0.0

    jerk_gt = x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]
    jerk_pr = x_hat[:, 3:] - 3 * x_hat[:, 2:-1] + 3 * x_hat[:, 1:-2] - x_hat[:, :-3]
    return masked_mse(jerk_gt, jerk_pr, jerk_mask)


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


def select_keyframe_indices(
    length: int,
    keyframe_interval: int,
    strategy: str,
    keyframe_count: int | None,
    keyframe_min: int,
    keyframe_max: int,
    include_ends: bool,
) -> list[int]:
    if length <= 0:
        return [0]

    if strategy == 'random':
        if keyframe_count is not None:
            k = keyframe_count
        else:
            k = random.randint(keyframe_min, keyframe_max)

        if include_ends:
            k = max(k, 2)

        k = max(1, min(k, length))

        if include_ends and length >= 2:
            indices = {0, length - 1}
            remaining = [i for i in range(1, length - 1)]
            k_remaining = max(0, k - len(indices))
            if k_remaining > 0 and len(remaining) > 0:
                indices.update(random.sample(remaining, min(k_remaining, len(remaining))))
        else:
            indices = set(random.sample(range(length), k))

        return sorted(indices)

    # interval strategy (default)
    indices = list(range(0, length, keyframe_interval))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def load_inbetween_model(
    cfg,
    device: str = 'cuda',
    inbetween_ckpt_path: str | None = None,
):
    """Load external CondMDI runtime and normalization stats from a checkpoint."""
    from models.selector_modules import build_keyframe_selector

    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    Fdim = mean.shape[0]

    if inbetween_ckpt_path is None:
        inbetween_ckpt_path = f'checkpoints/composite_inbetween_step{cfg.inbetween_steps}.pt'

    if looks_like_condmdi_checkpoint(inbetween_ckpt_path):
        inbetween_model, diff_inbetween = load_external_condmdi_runtime(
            checkpoint_path=inbetween_ckpt_path,
            local_mean=mean,
            local_std=std,
            device=device,
        )
        clip_model = setup_clip_model(device)
        print(f'Loaded external CondMDI runtime from {inbetween_ckpt_path}')
        return inbetween_model, diff_inbetween, clip_model, mean, std, Fdim

    ckpt = None
    if os.path.exists(inbetween_ckpt_path):
        ckpt = torch.load(inbetween_ckpt_path, map_location=device)

    selector_state = ckpt.get('selector_ema', ckpt.get('selector')) if isinstance(ckpt, dict) else None
    saved_cfg = ckpt.get('cfg') if isinstance(ckpt, dict) else None
    external_inbetween_ckpt_path = saved_cfg.get('external_inbetween_ckpt_path') if isinstance(saved_cfg, dict) else None

    def _count_prefix_layers(state_dict: dict, prefix: str, index_pos: int) -> int:
        idx = set()
        for k in state_dict.keys():
            if not k.startswith(prefix):
                continue
            parts = k.split('.')
            if len(parts) > index_pos and parts[index_pos].isdigit():
                idx.add(int(parts[index_pos]))
        return (max(idx) + 1) if idx else 0

    def _attach_selector(runtime_model, selector_state_dict, saved_cfg_dict):
        if not (cfg.use_learned_keyframe_selector and selector_state_dict is not None):
            return runtime_model

        saved_selector_mode = cfg.selector_mode
        saved_selector_heads = cfg.selector_heads
        saved_selector_threshold = cfg.selector_threshold
        saved_selector_ratio = cfg.selector_target_ratio
        if isinstance(saved_cfg_dict, dict):
            saved_selector_mode = str(saved_cfg_dict.get('selector_mode', saved_selector_mode))
            saved_selector_heads = int(saved_cfg_dict.get('selector_heads', saved_selector_heads))
            saved_selector_threshold = float(saved_cfg_dict.get('selector_threshold', saved_selector_threshold))
            saved_selector_ratio = float(saved_cfg_dict.get('selector_target_ratio', saved_selector_ratio))

        selector_d_model = cfg.selector_d_model
        if isinstance(selector_state_dict, dict) and 'frame_in.weight' in selector_state_dict:
            selector_d_model = int(selector_state_dict['frame_in.weight'].shape[0])
        selector_layers = cfg.selector_layers
        if isinstance(selector_state_dict, dict):
            selector_layers = _count_prefix_layers(selector_state_dict, 'encoder.layers.', 2) or selector_layers

        selector_model = build_keyframe_selector(
            mode=saved_selector_mode,
            feature_dim=Fdim,
            cond_dim=512,
            d_model=selector_d_model,
            n_layers=selector_layers,
            n_heads=saved_selector_heads,
            dropout=cfg.selector_dropout,
            max_len=cfg.max_len + 10,
            threshold=saved_selector_threshold,
            budget_ratio=saved_selector_ratio,
        ).to(device)
        if isinstance(selector_state_dict, dict) and len(selector_state_dict) > 0:
            selector_model.load_state_dict(selector_state_dict, strict=False)
        selector_model.eval()
        runtime_model.keyframe_selector = selector_model
        print(f'Loaded {saved_selector_mode} keyframe selector from checkpoint (EMA if available)')
        return runtime_model

    if external_inbetween_ckpt_path:
        resolved_external_path = os.path.abspath(os.path.join(os.path.dirname(inbetween_ckpt_path), external_inbetween_ckpt_path))
        inbetween_model, diff_inbetween = load_external_condmdi_runtime(
            checkpoint_path=resolved_external_path,
            local_mean=mean,
            local_std=std,
            device=device,
        )
        inbetween_model = _attach_selector(inbetween_model, selector_state, saved_cfg)
        inbetween_model.eval()
        clip_model = setup_clip_model(device)
        print(f'Loaded external CondMDI runtime from selector checkpoint {inbetween_ckpt_path}')
        return inbetween_model, diff_inbetween, clip_model, mean, std, Fdim

    raise ValueError(
        'Local diffusion checkpoints are no longer supported. '
        'Pass an external CondMDI checkpoint directly, or use a selector checkpoint '
        'whose cfg.external_inbetween_ckpt_path points to a CondMDI checkpoint.'
    )
