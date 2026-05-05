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
    from keyframe_selectors import build_keyframe_selector

    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    Fdim = mean.shape[0]

    if inbetween_ckpt_path is None:
        raise ValueError('load_inbetween_model requires an explicit CondMDI or selector checkpoint path.')

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

    if not os.path.exists(inbetween_ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {inbetween_ckpt_path}')

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
