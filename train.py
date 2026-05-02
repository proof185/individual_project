"""Training script for composite motion generation."""

import argparse
import csv
import math
import os
import random
import time
from datetime import datetime
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from config import CompositeConfig
from dataset import HUMANML3DCompositeDataset, collate_composite
from models.vqvae import MotionVQVAE
from models.gpt import MotionGPT
from models.diffusion import InbetweenDiffusion, InbetweenTransformer, KeyframeSelector
from utils import (
    encode_text,
    encode_text_with_tokens,
    masked_mse,
    prepare_gpt_batch,
    setup_clip_model,
    vel_loss,
    weighted_masked_mse,
)


class TextEncoderWrapper:
    """Wrapper for text encoder that can be pickled for multiprocessing."""
    def __init__(self, clip_model):
        self.clip_model = clip_model
    
    def __call__(self, texts, normalize=True):
        return encode_text(self.clip_model, texts, normalize)

    def encode_with_tokens(self, texts, normalize=True):
        return encode_text_with_tokens(self.clip_model, texts, normalize)


def _count_prefix_layers(state_dict: dict, prefix: str, index_pos: int) -> int:
    idx = set()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        parts = key.split('.')
        if len(parts) > index_pos and parts[index_pos].isdigit():
            idx.add(int(parts[index_pos]))
    return (max(idx) + 1) if idx else 0


def _apply_inbetween_arch_from_checkpoint(cfg, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get('cfg') if isinstance(checkpoint, dict) else None
    state = checkpoint.get('inbetween') if isinstance(checkpoint, dict) else None
    selector_state = checkpoint.get('selector') if isinstance(checkpoint, dict) else None

    if isinstance(saved_cfg, dict):
        cfg.inbetween_d_model = int(saved_cfg.get('inbetween_d_model', saved_cfg.get('d_model', cfg.inbetween_d_model)))
        cfg.inbetween_layers = int(saved_cfg.get('inbetween_layers', saved_cfg.get('n_layers', cfg.inbetween_layers)))
        cfg.inbetween_heads = int(saved_cfg.get('inbetween_heads', saved_cfg.get('n_heads', cfg.inbetween_heads)))
        cfg.selector_d_model = int(saved_cfg.get('selector_d_model', cfg.selector_d_model))
        cfg.selector_layers = int(saved_cfg.get('selector_layers', cfg.selector_layers))
        cfg.selector_heads = int(saved_cfg.get('selector_heads', cfg.selector_heads))

    if isinstance(state, dict):
        if 'frame_in.weight' in state:
            cfg.inbetween_d_model = int(state['frame_in.weight'].shape[0])
        counted = _count_prefix_layers(state, 'encoder.layers.', 2)
        if counted > 0:
            cfg.inbetween_layers = counted

    if isinstance(selector_state, dict):
        if 'frame_in.weight' in selector_state:
            cfg.selector_d_model = int(selector_state['frame_in.weight'].shape[0])
        counted = _count_prefix_layers(selector_state, 'encoder.layers.', 2)
        if counted > 0:
            cfg.selector_layers = counted


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().item()
    try:
        return float(value)
    except Exception:
        return None


def _init_metrics_logger(stage_name: str, columns: list[str]):
    os.makedirs('training_logs', exist_ok=True)
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join('training_logs', f'{stage_name}_{run_stamp}.csv')
    plot_path = os.path.join('training_logs', f'{stage_name}_{run_stamp}.png')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

    return {
        'stage_name': stage_name,
        'columns': columns,
        'csv_path': csv_path,
        'plot_path': plot_path,
        'rows': [],
    }


def _append_metrics_row(logger, row: dict):
    out = {col: row.get(col, '') for col in logger['columns']}
    with open(logger['csv_path'], 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=logger['columns'])
        writer.writerow(out)
    logger['rows'].append(out)


def _save_convergence_plot(logger):
    if plt is None:
        return

    if not logger['rows']:
        return

    numeric_cols = [c for c in logger['columns'] if c != 'step']
    if not numeric_cols:
        return

    series = {c: [] for c in numeric_cols}
    steps = []
    for row in logger['rows']:
        step = _safe_float(row.get('step'))
        if step is None:
            continue

        has_any = False
        parsed = {}
        for col in numeric_cols:
            val = _safe_float(row.get(col))
            parsed[col] = val
            if val is not None:
                has_any = True

        if not has_any:
            continue

        steps.append(step)
        for col in numeric_cols:
            series[col].append(parsed[col])

    if not steps:
        return

    plt.figure(figsize=(10, 6))
    for col in numeric_cols:
        col_vals = series[col]
        if not col_vals or all(v is None for v in col_vals):
            continue
        ys = [np.nan if v is None else v for v in col_vals]
        plt.plot(steps, ys, label=col)

    plt.title(f"{logger['stage_name']} convergence")
    plt.xlabel('step')
    plt.ylabel('metric value')
    plt.grid(alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(logger['plot_path'], dpi=160)
    plt.close()


def _lr_scale(step: int, total_steps: int, warmup_ratio: float, min_lr_ratio: float, scheduler_type: str) -> float:
    if scheduler_type == 'constant' or total_steps <= 0:
        return 1.0

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step <= warmup_steps:
        return max(1e-8, float(step) / float(warmup_steps))

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _apply_lr_schedule(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    step: int,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
    scheduler_type: str,
):
    scale = _lr_scale(step, total_steps, warmup_ratio, min_lr_ratio, scheduler_type)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg['lr'] = float(base_lr) * scale


def _clone_state_dict(module: nn.Module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _update_ema_state(ema_state: dict[str, torch.Tensor], module: nn.Module, decay: float):
    with torch.no_grad():
        current = module.state_dict()
        for k, v in current.items():
            if k not in ema_state:
                ema_state[k] = v.detach().clone()
                continue
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.detach(), alpha=(1.0 - decay))
            else:
                ema_state[k].copy_(v)


def _build_cond(batch, dataset, empty_emb, p_uncond: float):
    cond_list = []
    for mid, tidx in zip(batch['ids'], batch['text_idxs']):
        if random.random() < p_uncond:
            cond_list.append(empty_emb)
        else:
            cond_list.append(dataset.get_embedding(mid, tidx))
    return torch.stack(cond_list)


def _compute_inbetween_loss(
    inbetween_model,
    diff_inbetween,
    x0,
    mask,
    keyframes,
    keyframe_indices,
    keyframe_mask,
    cond,
    cfg,
    keyframe_selector,
    selector_budget_weight: float,
    selector_entropy_weight: float,
):
    B = x0.shape[0]
    t = torch.randint(0, cfg.T_diffusion, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = diff_inbetween.q_sample(x0, t, noise)
    xt = xt * mask.float().unsqueeze(-1)

    selector_ratio = None
    selector_budget_loss = x0.new_tensor(0.0)
    selector_entropy_loss = x0.new_tensor(0.0)

    if keyframe_selector is not None:
        selector_probs, selector_mask_st = keyframe_selector(x0, mask, cond=cond)
        keyframe_canvas = selector_mask_st.unsqueeze(-1) * x0

        xt = diff_inbetween._replace_keyframes(
            xt,
            observation_mask=selector_mask_st,
            keyframe_canvas=keyframe_canvas,
        )
        x0_hat = inbetween_model(
            xt,
            t,
            cond,
            mask,
            observation_mask=selector_mask_st,
            keyframe_canvas=keyframe_canvas,
        )

        non_keyframe_weights = (1.0 - selector_mask_st) * mask.float()
        loss = weighted_masked_mse(x0, x0_hat, non_keyframe_weights, mask)

        selector_ratio = (selector_probs * mask.float()).sum() / (mask.float().sum() + 1e-8)
        selector_budget_loss = (selector_ratio - cfg.selector_target_ratio) ** 2
        p = selector_probs.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        selector_entropy_loss = (entropy * mask.float()).sum() / (mask.float().sum() + 1e-8)
        loss = (
            loss
            + selector_budget_weight * selector_budget_loss
            + selector_entropy_weight * selector_entropy_loss
        )

        keyframes, keyframe_indices, keyframe_mask = _frame_mask_to_sparse_keyframes(
            x0,
            selector_mask_st.detach(),
            mask,
        )
    else:
        xt = diff_inbetween._replace_keyframes(xt, keyframes, keyframe_indices, keyframe_mask)
        x0_hat = inbetween_model(xt, t, cond, mask, keyframes, keyframe_indices, keyframe_mask)
        non_keyframe_mask = mask.clone()
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            non_keyframe_mask[b, valid_idx] = False
        loss = masked_mse(x0, x0_hat, non_keyframe_mask)

    return {
        'loss': loss,
        'selector_ratio': selector_ratio,
        'selector_budget': selector_budget_loss,
        'selector_entropy': selector_entropy_loss,
    }


@torch.no_grad()
def _evaluate_inbetween(
    inbetween_model,
    keyframe_selector,
    diff_inbetween,
    val_loader,
    dataset,
    empty_emb,
    cfg,
    device,
):
    inbetween_model.eval()
    if keyframe_selector is not None:
        keyframe_selector.eval()

    loss_values = []
    max_batches = max(1, int(cfg.val_batches))
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x0 = batch['motion'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        keyframes = batch['keyframes'].to(device, non_blocking=True)
        keyframe_indices = batch['keyframe_indices'].to(device, non_blocking=True)
        keyframe_mask = batch['keyframe_mask'].to(device, non_blocking=True)
        cond = _build_cond(batch, dataset, empty_emb, p_uncond=0.0)

        out = _compute_inbetween_loss(
            inbetween_model,
            diff_inbetween,
            x0,
            mask,
            keyframes,
            keyframe_indices,
            keyframe_mask,
            cond,
            cfg,
            keyframe_selector,
            selector_budget_weight=cfg.selector_budget_weight,
            selector_entropy_weight=cfg.selector_entropy_weight,
        )
        loss_values.append(out['loss'].item())

    inbetween_model.train()
    if keyframe_selector is not None:
        keyframe_selector.train()

    if not loss_values:
        return None
    return float(sum(loss_values) / len(loss_values))


def main(
    stage: str,
    force_retrain: bool,
    vqvae_steps: int | None = None,
    gpt_steps: int | None = None,
    inbetween_steps: int | None = None,
    inbetween_resume: str | None = None,
    keyframe_source_dir: str | None = None,
    disable_selector: bool = False,
    lr: float | None = None,
    inbetween_ckpt_prefix: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    scheduler_type: str | None = None,
    warmup_ratio: float | None = None,
    min_lr_ratio: float | None = None,
    inbetween_lr: float | None = None,
    selector_lr: float | None = None,
    ema_decay: float | None = None,
    selector_curriculum_fraction: float | None = None,
    val_interval: int | None = None,
    val_batches: int | None = None,
    grad_accum_steps: int | None = None,
):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    cfg = CompositeConfig(
        keyframe_strategy='random',      # Switch to random
        keyframe_min=3,                  # Min keyframes per sequence
        keyframe_max=20,                 # Max keyframes per sequence
        keyframe_include_ends=True,      # Always include first/last frame
        # ... other params
    )
    if vqvae_steps is not None:
        cfg.vqvae_steps = int(vqvae_steps)
    if gpt_steps is not None:
        cfg.gpt_steps = int(gpt_steps)
    if inbetween_steps is not None:
        cfg.inbetween_steps = int(inbetween_steps)
    if inbetween_resume is not None:
        inbetween_resume = inbetween_resume.strip()
    if keyframe_source_dir is not None:
        cfg.keyframe_source_dir = keyframe_source_dir
    if disable_selector:
        cfg.use_learned_keyframe_selector = False
    if lr is not None:
        cfg.lr = float(lr)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if num_workers is not None:
        cfg.num_workers = int(num_workers)
    if scheduler_type is not None:
        cfg.scheduler_type = scheduler_type
    if warmup_ratio is not None:
        cfg.warmup_ratio = float(warmup_ratio)
    if min_lr_ratio is not None:
        cfg.min_lr_ratio = float(min_lr_ratio)
    if inbetween_lr is not None:
        cfg.inbetween_lr = float(inbetween_lr)
    if selector_lr is not None:
        cfg.selector_lr = float(selector_lr)
    if ema_decay is not None:
        cfg.ema_decay = float(ema_decay)
    if selector_curriculum_fraction is not None:
        cfg.selector_curriculum_fraction = float(selector_curriculum_fraction)
    if val_interval is not None:
        cfg.val_interval = int(val_interval)
    if val_batches is not None:
        cfg.val_batches = int(val_batches)
    if grad_accum_steps is not None:
        cfg.grad_accum_steps = max(1, int(grad_accum_steps))
    if inbetween_ckpt_prefix is None or not inbetween_ckpt_prefix.strip():
        inbetween_ckpt_prefix = 'composite_inbetween'
    inbetween_ckpt_prefix = inbetween_ckpt_prefix.strip()
    if stage in {'all', 'inbetween'}:
        selector_state = 'enabled' if cfg.use_learned_keyframe_selector else 'disabled'
        print(f"Keyframe strategy (dataset fallback): {cfg.keyframe_strategy} | learned selector: {selector_state}")
    else:
        print(f"Keyframe strategy: {cfg.keyframe_strategy}")

    inbetween_final_ckpt_path = f'checkpoints/{inbetween_ckpt_prefix}_step{cfg.inbetween_steps}.pt'
    if stage in {'all', 'inbetween'} and not force_retrain:
        arch_ckpt_path = None
        if inbetween_resume is not None and os.path.exists(inbetween_resume):
            arch_ckpt_path = inbetween_resume
        elif os.path.exists(inbetween_final_ckpt_path):
            arch_ckpt_path = inbetween_final_ckpt_path
        if arch_ckpt_path is not None:
            _apply_inbetween_arch_from_checkpoint(cfg, arch_ckpt_path, device)
            print(
                'In-between architecture from checkpoint:',
                arch_ckpt_path,
                f"d_model={cfg.inbetween_d_model}",
                f"layers={cfg.inbetween_layers}",
                f"selector_d_model={cfg.selector_d_model}",
                f"selector_layers={cfg.selector_layers}",
            )
    
    # Load normalization statistics
    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    Fdim = mean.shape[0]
    print('Feature dim:', Fdim)
    
    # Setup CLIP
    clip_model = setup_clip_model(device)
    
    text_encoder = TextEncoderWrapper(clip_model)
    
    empty_emb, empty_token_features, empty_token_mask = text_encoder.encode_with_tokens([''])
    empty_emb = empty_emb.squeeze(0)
    empty_token_features = empty_token_features.squeeze(0)
    empty_token_mask = empty_token_mask.squeeze(0)
    print('Empty embedding norm:', empty_emb.norm().item())
    
    # Create dataset
    dataset = HUMANML3DCompositeDataset(
        cfg.root,
        split='train',
        max_len=cfg.max_len,
        normalize=True,
        use_cache=True,
        text_encoder=text_encoder,
        keyframe_interval=cfg.keyframe_interval,
        keyframe_strategy=cfg.keyframe_strategy,
        keyframe_count=cfg.keyframe_count,
        keyframe_min=cfg.keyframe_min,
        keyframe_max=cfg.keyframe_max,
        keyframe_include_ends=cfg.keyframe_include_ends,
        include_keyframes=(stage != 'gpt'),
        conditioning_motion_dir=cfg.keyframe_source_dir,
        mean=mean,
        std=std,
        device=device,
    )
    print('Dataset size:', len(dataset))

    val_dataset = None
    val_loader = None
    if stage in {'all', 'inbetween'}:
        val_split_file = os.path.join(cfg.root, f'{cfg.val_split}.txt')
        if os.path.exists(val_split_file):
            val_dataset = HUMANML3DCompositeDataset(
                cfg.root,
                split=cfg.val_split,
                max_len=cfg.max_len,
                normalize=True,
                use_cache=True,
                text_encoder=text_encoder,
                keyframe_interval=cfg.keyframe_interval,
                keyframe_strategy=cfg.keyframe_strategy,
                keyframe_count=cfg.keyframe_count,
                keyframe_min=cfg.keyframe_min,
                keyframe_max=cfg.keyframe_max,
                keyframe_include_ends=cfg.keyframe_include_ends,
                conditioning_motion_dir=cfg.keyframe_source_dir,
                mean=mean,
                std=std,
                device=device,
            )
            val_loader_kwargs = {
                'batch_size': min(cfg.val_batch_size, cfg.batch_size),
                'shuffle': False,
                'num_workers': cfg.num_workers,
                'collate_fn': collate_composite,
                'drop_last': False,
                'pin_memory': bool(cfg.pin_memory and device.type == 'cuda'),
            }
            if cfg.num_workers > 0:
                val_loader_kwargs['persistent_workers'] = cfg.persistent_workers
                val_loader_kwargs['prefetch_factor'] = cfg.prefetch_factor
            val_loader = DataLoader(val_dataset, **val_loader_kwargs)
            print(f"Validation dataset size: {len(val_dataset)} ({cfg.val_split})")
        else:
            print(f"Validation split file not found: {val_split_file}; best-checkpoint selection disabled.")
    
    loader_kwargs = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'collate_fn': collate_composite,
        'drop_last': True,
        'pin_memory': bool(cfg.pin_memory and device.type == 'cuda'),
    }
    if cfg.num_workers > 0:
        loader_kwargs['persistent_workers'] = cfg.persistent_workers
        loader_kwargs['prefetch_factor'] = cfg.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    print(
        'DataLoader:',
        f"workers={cfg.num_workers}",
        f"batch_size={cfg.batch_size}",
        f"grad_accum={cfg.grad_accum_steps}",
        f"effective_batch={cfg.batch_size * cfg.grad_accum_steps}",
        f"pin_memory={loader_kwargs['pin_memory']}",
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)}",
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}"
    )
    
    # Initialize only models needed for selected stage(s).
    vqvae = None
    gpt = None
    if stage in {'all', 'vqvae', 'gpt'}:
        vqvae = MotionVQVAE(
            feature_dim=Fdim,
            codebook_size=cfg.codebook_size,
            codebook_dim=cfg.codebook_dim,
            downsample_rate=cfg.downsample_rate,
            commitment_cost=cfg.commitment_cost,
        ).to(device)
        print(f'VQ-VAE params: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M')

    if stage in {'all', 'gpt'}:
        gpt = MotionGPT(
            codebook_size=cfg.codebook_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            max_len=cfg.max_len // cfg.downsample_rate + 10,
            cond_dim=512,
        ).to(device)
        print(f'MotionGPT params: {sum(p.numel() for p in gpt.parameters())/1e6:.2f}M')
    elif stage == 'inbetween':
        print('Skipping VQ-VAE and MotionGPT initialization (stage=inbetween).')
    
    inbetween_model = InbetweenTransformer(
        feature_dim=Fdim,
        cond_dim=512,
        d_model=cfg.inbetween_d_model,
        n_layers=cfg.inbetween_layers,
        n_heads=cfg.inbetween_heads,
        dropout=cfg.dropout,
        max_len=cfg.max_len + 10,
    ).to(device)
    print(f'In-betweening model params: {sum(p.numel() for p in inbetween_model.parameters())/1e6:.2f}M')

    keyframe_selector = None
    if cfg.use_learned_keyframe_selector:
        keyframe_selector = KeyframeSelector(
            feature_dim=Fdim,
            cond_dim=512,
            d_model=cfg.selector_d_model,
            n_layers=cfg.selector_layers,
            n_heads=cfg.selector_heads,
            dropout=cfg.selector_dropout,
            max_len=cfg.max_len + 10,
            threshold=cfg.selector_threshold,
        ).to(device)
        print(f'Keyframe selector params: {sum(p.numel() for p in keyframe_selector.parameters())/1e6:.2f}M')
    
    diff_inbetween = InbetweenDiffusion(cfg.T_diffusion, device=device)
    
    # Stage 1a: Train VQ-VAE
    if stage in {'all', 'vqvae'}:
        if vqvae is None:
            raise RuntimeError('VQ-VAE model is not initialized for requested stage.')
        train_vqvae(vqvae, loader, cfg, device, mean, std, force_retrain)
    
    # Stage 1b: Train GPT
    if stage in {'all', 'gpt'}:
        
        if vqvae is None or gpt is None:
            raise RuntimeError('VQ-VAE/GPT models are not initialized for requested stage.')
        
        vqvae_ckpt_path = f'checkpoints/composite_vqvae_step{cfg.vqvae_steps}.pt'
        if not os.path.exists(vqvae_ckpt_path):
            raise FileNotFoundError(f"Missing VQ-VAE checkpoint: {vqvae_ckpt_path}")

        print(f"Loading trained VQ-VAE for GPT tokenisation: {vqvae_ckpt_path}")
        vqvae_ckpt = torch.load(vqvae_ckpt_path, map_location=device)
        vqvae.load_state_dict(vqvae_ckpt['model'])
        vqvae.eval()
        
        train_gpt(
            vqvae,
            gpt,
            loader,
            dataset,
            empty_token_features,
            empty_token_mask,
            cfg,
            device,
            force_retrain,
        )
    
    # Stage 2: Train Diffusion In-betweening
    if stage in {'all', 'inbetween'}:
        train_inbetween(
            inbetween_model,
            diff_inbetween,
            loader,
            val_loader,
            dataset,
            empty_emb,
            cfg,
            device,
            mean,
            std,
            force_retrain,
            keyframe_selector=keyframe_selector,
            resume_ckpt_path=inbetween_resume,
            ckpt_prefix=inbetween_ckpt_prefix,
        )
    
    print("Training complete!")


def train_vqvae(vqvae, loader, cfg, device, mean, std, force_retrain: bool = False):
    """Stage 1a: Train VQ-VAE."""
    vqvae_final_ckpt_path = f'checkpoints/composite_vqvae_step{cfg.vqvae_steps}.pt'
    if os.path.exists(vqvae_final_ckpt_path) and not force_retrain:
        print(f"Loading VQ-VAE from final checkpoint: {vqvae_final_ckpt_path}")
        vqvae_ckpt = torch.load(vqvae_final_ckpt_path, map_location=device)
        vqvae.load_state_dict(vqvae_ckpt['model'])
        print("VQ-VAE training already complete! Loaded from checkpoint.")
        return
    
    vqvae_optimizer = torch.optim.AdamW(vqvae.parameters(), lr=cfg.lr)
    base_lrs = [cfg.lr]
    lambda_recon = 1.0
    lambda_vel = 0.5
    metrics_logger = _init_metrics_logger(
        'vqvae',
        ['step', 'recon_loss', 'vq_loss', 'vel_loss', 'total_loss']
    )
    print(f"VQ-VAE metrics CSV: {metrics_logger['csv_path']}")
    
    vqvae.train()
    data_iter = cycle(loader)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    
    print("Stage 1a: Training VQ-VAE...")
    start_time = time.time()
    
    for step in range(1, cfg.vqvae_steps + 1):
        _apply_lr_schedule(
            vqvae_optimizer,
            base_lrs,
            step,
            cfg.vqvae_steps,
            cfg.warmup_ratio,
            cfg.min_lr_ratio,
            cfg.scheduler_type,
        )
        vqvae_optimizer.zero_grad(set_to_none=True)
        recon_running = 0.0
        vq_running = 0.0
        vel_running = 0.0
        total_running = 0.0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)
            x = batch['motion'].to(device)
            mask = batch['mask'].to(device)

            B, T, _ = x.shape
            pad_len = (cfg.downsample_rate - T % cfg.downsample_rate) % cfg.downsample_rate
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
                mask = F.pad(mask, (0, pad_len), value=False)

            x_recon, indices, vq_loss = vqvae(x, mask)

            x_recon = x_recon[:, :T]
            mask_orig = batch['mask'].to(device)

            recon_loss = masked_mse(batch['motion'].to(device), x_recon, mask_orig)
            v_loss = vel_loss(batch['motion'].to(device), x_recon, mask_orig)

            loss = lambda_recon * recon_loss + vq_loss + lambda_vel * v_loss
            (loss / grad_accum_steps).backward()

            recon_running += recon_loss.item()
            vq_running += vq_loss.item()
            vel_running += v_loss.item()
            total_running += loss.item()

        nn.utils.clip_grad_norm_(vqvae.parameters(), cfg.grad_clip)
        vqvae_optimizer.step()

        recon_avg = recon_running / grad_accum_steps
        vq_avg = vq_running / grad_accum_steps
        vel_avg = vel_running / grad_accum_steps
        total_avg = total_running / grad_accum_steps
        
        if step % 200 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            steps_done = step
            steps_remaining = cfg.vqvae_steps - step
            avg_time_per_step = elapsed / steps_done
            eta_seconds = steps_remaining * avg_time_per_step
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            lr_now = vqvae_optimizer.param_groups[0]['lr']
            print(f"step {step:>7d} | recon {recon_avg:.5f} | vq {vq_avg:.5f} | vel {vel_avg:.5f} | lr {lr_now:.2e} | ETA: {eta_hours}h {eta_minutes}m")
            _append_metrics_row(metrics_logger, {
                'step': step,
                'recon_loss': recon_avg,
                'vq_loss': vq_avg,
                'vel_loss': vel_avg,
                'total_loss': total_avg,
            })

        if step % 2_000 == 0:
            _save_convergence_plot(metrics_logger)
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model': vqvae.state_dict(),
                'optimizer': vqvae_optimizer.state_dict(),
                'step': step,
            }, f'checkpoints/composite_vqvae_step{step}.pt')
            print('Saved VQ-VAE checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model': vqvae.state_dict(),
        'optimizer': vqvae_optimizer.state_dict(),
        'step': cfg.vqvae_steps,
    }, vqvae_final_ckpt_path)
    print(f'Saved final VQ-VAE checkpoint: {vqvae_final_ckpt_path}')
    _save_convergence_plot(metrics_logger)
    print(f"VQ-VAE convergence plot: {metrics_logger['plot_path']}")
    
    print("VQ-VAE training complete!")


# ---------------------------------------------------------------------------
# GPT pre-tokenization helpers
# ---------------------------------------------------------------------------

def _pretokenize_dataset(vqvae, dataset, cfg, device):
    """Run VQ-VAE encode over every dataset motion once and cache the tokens.

    VQ tokens are deterministic (argmax), so running the encoder on every GPT
    training step is pure overhead.  This function computes the tokens once,
    saves them to disk, and returns a dict ``{sample_id: (tokens_cpu, token_len)}``.
    """
    cache_path = os.path.join(
        cfg.root,
        f'gpt_tokens_cb{cfg.codebook_size}_ds{cfg.downsample_rate}_maxlen{cfg.max_len}.pt',
    )
    if os.path.exists(cache_path):
        print(f'Loading pre-tokenised GPT data from {cache_path}')
        return torch.load(cache_path, map_location='cpu')

    print('Pre-tokenising dataset with VQ-VAE (one-time cost, result cached)...')
    vqvae.eval()
    tokens_dict = {}
    n = len(dataset.data)
    with torch.no_grad():
        for i, item in enumerate(dataset.data):
            mid = item['id']
            motion = item['motion']          # (T, F) normalised CPU tensor
            actual_length = int(item['length'])
            T = motion.shape[0]

            x = motion.unsqueeze(0).to(device)
            pad_len = (cfg.downsample_rate - T % cfg.downsample_rate) % cfg.downsample_rate
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))

            indices, _ = vqvae.encode(x)          # (1, T_tokens)
            tok_len = min(
                (actual_length + cfg.downsample_rate - 1) // cfg.downsample_rate,
                indices.shape[1],
            )
            tokens_dict[mid] = (indices[0].cpu(), tok_len)

            if (i + 1) % 1000 == 0 or (i + 1) == n:
                print(f'  tokenised {i + 1}/{n} sequences')

    torch.save(tokens_dict, cache_path)
    print(f'Saved token cache: {cache_path}')
    return tokens_dict


def _build_gpt_batch_from_tokens(batch, tokens_dict, gpt, device):
    """Build GPT input/target/mask tensors from pre-tokenised data.

    Replaces ``prepare_gpt_batch`` in the hot loop, eliminating the per-step
    VQ-VAE encode.
    """
    ids = batch['ids']
    B = len(ids)

    token_seqs = []
    token_lengths = []
    for mid in ids:
        tokens, tok_len = tokens_dict[mid]
        token_seqs.append(tokens[:tok_len])
        token_lengths.append(tok_len)

    T_max = max(token_lengths)

    input_tokens = torch.full(
        (B, T_max + 1),
        gpt.pad_token,
        dtype=torch.long,
        device=device,
    )
    target_tokens = torch.full(
        (B, T_max + 1),
        gpt.pad_token,
        dtype=torch.long,
        device=device,
    )
    target_mask = torch.zeros(
        B,
        T_max + 1,
        dtype=torch.bool,
        device=device,
    )

    for i, (seq, tlen) in enumerate(zip(token_seqs, token_lengths)):
        seq = seq[:tlen].to(device, non_blocking=True)

        input_tokens[i, 0] = gpt.bos_token
        input_tokens[i, 1:tlen + 1] = seq

        target_tokens[i, :tlen] = seq
        target_tokens[i, tlen] = gpt.eos_token

        target_mask[i, :tlen + 1] = True

    return input_tokens, target_tokens, target_mask


def _build_gpt_cond_batch(batch, dataset, empty_tokens, empty_mask, p_uncond: float, device):
    cond_list = []
    mask_list = []
    for mid, tidx in zip(batch['ids'], batch['text_idxs']):
        if random.random() < p_uncond:
            cond_list.append(empty_tokens)
            mask_list.append(empty_mask)
        else:
            cond_list.append(dataset.get_token_embedding(mid, tidx))
            mask_list.append(dataset.get_token_mask(mid, tidx))

    max_ctx = max(int(mask.shape[0]) for mask in mask_list)
    feat_dim = int(cond_list[0].shape[-1])
    cond = torch.zeros(len(cond_list), max_ctx, feat_dim, dtype=cond_list[0].dtype, device=device)
    cond_mask = torch.zeros(len(cond_list), max_ctx, dtype=torch.bool, device=device)

    for i, (feat, mask) in enumerate(zip(cond_list, mask_list)):
        ctx_len = int(mask.shape[0])
        cond[i, :ctx_len] = feat.to(device, non_blocking=True)
        cond_mask[i, :ctx_len] = mask.to(device, non_blocking=True)

    return cond, cond_mask


def train_gpt(vqvae, gpt, loader, dataset, empty_cond_tokens, empty_cond_mask, cfg, device, force_retrain: bool = False):
    """Stage 1b: Train MotionGPT."""
    gpt_final_ckpt_path = f'checkpoints/composite_gpt_step{cfg.gpt_steps}.pt'
    if os.path.exists(gpt_final_ckpt_path) and not force_retrain:
        print(f"Loading GPT from final checkpoint: {gpt_final_ckpt_path}")
        gpt_ckpt = torch.load(gpt_final_ckpt_path, map_location=device)
        try:
            gpt.load_state_dict(gpt_ckpt['gpt'])
        except RuntimeError as exc:
            raise RuntimeError(
                'Existing GPT checkpoint is incompatible with the new token-level text conditioning. '
                'Retrain GPT with --force to use the updated architecture.'
            ) from exc
        vqvae.load_state_dict(gpt_ckpt['vqvae'])
        print("MotionGPT training already complete! Loaded from checkpoint.")
        return
    
    vqvae.eval()
    gpt_optimizer = torch.optim.AdamW(gpt.parameters(), lr=cfg.lr)
    base_lrs = [cfg.lr]
    metrics_logger = _init_metrics_logger('gpt', ['step', 'loss', 'ppl'])
    print(f"GPT metrics CSV: {metrics_logger['csv_path']}")

    # Pre-tokenise the entire dataset once so the VQ-VAE encode is not
    # repeated on every training step (it's deterministic, so pure overhead).
    tokens_dict = _pretokenize_dataset(vqvae, dataset, cfg, device)

    gpt.train()
    data_iter = cycle(loader)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    
    print("Stage 1b: Training MotionGPT...")
    start_time = time.time()
    
    for step in range(1, cfg.gpt_steps + 1):
        _apply_lr_schedule(
            gpt_optimizer,
            base_lrs,
            step,
            cfg.gpt_steps,
            cfg.warmup_ratio,
            cfg.min_lr_ratio,
            cfg.scheduler_type,
        )
        gpt_optimizer.zero_grad(set_to_none=True)
        loss_running = 0.0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)

            input_tokens, target_tokens, target_mask = _build_gpt_batch_from_tokens(
                batch, tokens_dict, gpt, device
            )

            cond, cond_mask = _build_gpt_cond_batch(
                batch,
                dataset,
                empty_cond_tokens,
                empty_cond_mask,
                cfg.p_uncond,
                device,
            )

            logits = gpt(input_tokens, cond, cond_mask=cond_mask)

            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                target_tokens.reshape(-1),
                ignore_index=gpt.pad_token,
                reduction='none'
            ).reshape(B, T)

            loss = (loss * target_mask.float()).sum() / (target_mask.float().sum() + 1e-8)
            (loss / grad_accum_steps).backward()
            loss_running += loss.item()

        nn.utils.clip_grad_norm_(gpt.parameters(), cfg.grad_clip)
        gpt_optimizer.step()

        loss_avg = loss_running / grad_accum_steps
        
        if step % 200 == 0:
            ppl = math.exp(loss_avg)
            current_time = time.time()
            elapsed = current_time - start_time
            steps_remaining = cfg.gpt_steps - step
            avg_time_per_step = elapsed / step
            eta_seconds = steps_remaining * avg_time_per_step
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            lr_now = gpt_optimizer.param_groups[0]['lr']
            print(f"step {step:>7d} | loss {loss_avg:.5f} | ppl {ppl:.2f} | lr {lr_now:.2e} | ETA: {eta_hours}h {eta_minutes}m")
            _append_metrics_row(metrics_logger, {
                'step': step,
                'loss': loss_avg,
                'ppl': ppl,
            })

        if step % 2_000 == 0:
            _save_convergence_plot(metrics_logger)
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'gpt': gpt.state_dict(),
                'vqvae': vqvae.state_dict(),
                'optimizer': gpt_optimizer.state_dict(),
                'step': step,
            }, f'checkpoints/composite_gpt_step{step}.pt')
            print('Saved GPT checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'gpt': gpt.state_dict(),
        'vqvae': vqvae.state_dict(),
        'optimizer': gpt_optimizer.state_dict(),
        'step': cfg.gpt_steps,
    }, gpt_final_ckpt_path)
    print(f'Saved final GPT checkpoint: {gpt_final_ckpt_path}')
    _save_convergence_plot(metrics_logger)
    print(f"GPT convergence plot: {metrics_logger['plot_path']}")
    
    print("MotionGPT training complete!")


def _frame_mask_to_sparse_keyframes(
    x0: torch.Tensor,
    frame_mask: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """Convert per-frame keyframe mask (B, T) into padded sparse keyframe tensors (vectorized)."""
    B, T, F = x0.shape
    hard_mask = (frame_mask > 0.5) & valid_mask  # (B, T) bool

    # Count keyframes per batch item; pad all to K_max
    counts = hard_mask.sum(dim=1).clamp(min=1)  # (B,) — at least one per item
    K_max = int(counts.max().item())

    keyframe_indices = torch.zeros(B, K_max, device=x0.device, dtype=torch.long)
    keyframe_mask_out = torch.zeros(B, K_max, device=x0.device, dtype=torch.bool)

    # Fill row-by-row using the already-computed hard_mask; work entirely on CPU
    # for the index-packing step (one pass, no per-element GPU sync).
    hard_mask_cpu = hard_mask.cpu()
    for b in range(B):
        idx = hard_mask_cpu[b].nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.long)
        K = idx.numel()
        keyframe_indices[b, :K] = idx.to(x0.device)
        keyframe_mask_out[b, :K] = True

    # Gather keyframe features in one batched op (no loops)
    idx_exp = keyframe_indices.unsqueeze(-1).expand(B, K_max, F)  # (B, K_max, F)
    keyframes = torch.gather(x0, 1, idx_exp)                      # (B, K_max, F)

    return keyframes, keyframe_indices, keyframe_mask_out


def train_inbetween(
    inbetween_model,
    diff_inbetween,
    loader,
    val_loader,
    dataset,
    empty_emb,
    cfg,
    device,
    mean,
    std,
    force_retrain: bool = False,
    keyframe_selector=None,
    resume_ckpt_path: str | None = None,
    ckpt_prefix: str = 'composite_inbetween',
):
    """Stage 2: Train Diffusion In-betweening."""
    inbetween_final_ckpt_path = f'checkpoints/{ckpt_prefix}_step{cfg.inbetween_steps}.pt'
    inbetween_best_ckpt_path = f'checkpoints/{ckpt_prefix}_best.pt'
    if os.path.exists(inbetween_final_ckpt_path) and not force_retrain and not resume_ckpt_path:
        print(f"Loading In-Betweening model from final checkpoint: {inbetween_final_ckpt_path}")
        inbetween_ckpt = torch.load(inbetween_final_ckpt_path, map_location=device)
        if cfg.use_ema_for_sampling and inbetween_ckpt.get('inbetween_ema') is not None:
            inbetween_model.load_state_dict(inbetween_ckpt['inbetween_ema'])
        else:
            inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
        if keyframe_selector is not None:
            selector_key = 'selector_ema' if cfg.use_ema_for_sampling and inbetween_ckpt.get('selector_ema') is not None else 'selector'
            if inbetween_ckpt.get(selector_key) is not None:
                keyframe_selector.load_state_dict(inbetween_ckpt[selector_key])
            else:
                print('Warning: selector enabled in config but no selector weights found in checkpoint.')
        print("Diffusion in-betweening training already complete! Loaded from checkpoint.")
        return
    if os.path.exists(inbetween_final_ckpt_path) and not force_retrain and resume_ckpt_path:
        print(
            f"Ignoring existing final checkpoint {inbetween_final_ckpt_path} because explicit resume was requested: {resume_ckpt_path}"
        )

    inbetween_lr = float(cfg.inbetween_lr if cfg.inbetween_lr is not None else cfg.lr)
    if keyframe_selector is not None:
        selector_lr = float(cfg.selector_lr if cfg.selector_lr is not None else (inbetween_lr * cfg.selector_lr_scale))
        param_groups = [
            {'params': list(inbetween_model.parameters()), 'lr': inbetween_lr, 'name': 'inbetween'},
            {'params': list(keyframe_selector.parameters()), 'lr': selector_lr, 'name': 'selector'},
        ]
        base_lrs = [inbetween_lr, selector_lr]
    else:
        param_groups = [{'params': list(inbetween_model.parameters()), 'lr': inbetween_lr, 'name': 'inbetween'}]
        base_lrs = [inbetween_lr]

    inbetween_optimizer = torch.optim.AdamW(param_groups)
    all_params = []
    for group in param_groups:
        all_params.extend(group['params'])

    ema_inbetween_state = _clone_state_dict(inbetween_model)
    ema_selector_state = _clone_state_dict(keyframe_selector) if keyframe_selector is not None else None

    metrics_logger = _init_metrics_logger(
        'inbetween',
        [
            'step',
            'total_loss',
            'val_loss',
            'lr_inbetween',
            'lr_selector',
            'selector_budget_weight',
            'selector_entropy_weight',
            'selector_ratio',
            'selector_budget',
            'selector_entropy',
        ],
    )
    print(f"In-between metrics CSV: {metrics_logger['csv_path']}")

    start_step = 0
    if resume_ckpt_path and os.path.exists(resume_ckpt_path) and not force_retrain:
        print(f"Resuming In-Betweening from checkpoint: {resume_ckpt_path}")
        inbetween_ckpt = torch.load(resume_ckpt_path, map_location=device)
        inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
        if inbetween_ckpt.get('inbetween_ema') is not None:
            ema_inbetween_state = {k: v.detach().clone() for k, v in inbetween_ckpt['inbetween_ema'].items()}
        if keyframe_selector is not None:
            if inbetween_ckpt.get('selector') is not None:
                keyframe_selector.load_state_dict(inbetween_ckpt['selector'])
                if inbetween_ckpt.get('selector_ema') is not None:
                    ema_selector_state = {k: v.detach().clone() for k, v in inbetween_ckpt['selector_ema'].items()}
            else:
                print('Warning: resume checkpoint has no selector state; selector will train from scratch.')
        if 'optimizer' in inbetween_ckpt:
            try:
                inbetween_optimizer.load_state_dict(inbetween_ckpt['optimizer'])
            except Exception as exc:
                print(f"Warning: failed to load optimizer state, continuing with fresh optimizer ({exc})")
        start_step = int(inbetween_ckpt.get('step', 0))
        print(f"Resume step: {start_step}")
        if cfg.inbetween_steps <= start_step:
            print("Requested in-betweening steps already reached; nothing to train.")
            return

    best_val_loss = float('inf')
    best_step = 0
    if os.path.exists(inbetween_best_ckpt_path) and not force_retrain:
        try:
            best_ckpt = torch.load(inbetween_best_ckpt_path, map_location='cpu')
            best_val_loss = float(best_ckpt.get('best_val_loss', float('inf')))
            best_step = int(best_ckpt.get('best_step', 0))
            print(f"Current best checkpoint: step={best_step}, val_loss={best_val_loss:.6f}")
        except Exception as exc:
            print(f"Warning: failed to read best checkpoint metadata ({exc})")

    inbetween_model.train()
    if keyframe_selector is not None:
        keyframe_selector.train()
    data_iter = cycle(loader)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    
    if keyframe_selector is not None:
        print(f"Stage 2: Training Diffusion In-Betweening with LEARNED keyframe selector (target ratio={cfg.selector_target_ratio})...")
    else:
        print(f"Stage 2: Training Diffusion In-Betweening (keyframe interval={cfg.keyframe_interval})...")
    
    start_time = time.time()

    for step in range(start_step + 1, cfg.inbetween_steps + 1):
        _apply_lr_schedule(
            inbetween_optimizer,
            base_lrs,
            step,
            cfg.inbetween_steps,
            cfg.warmup_ratio,
            cfg.min_lr_ratio,
            cfg.scheduler_type,
        )
        inbetween_optimizer.zero_grad(set_to_none=True)

        if keyframe_selector is not None:
            curriculum_den = max(1.0, float(cfg.inbetween_steps) * float(cfg.selector_curriculum_fraction))
            curriculum_scale = min(1.0, max(0.0, float(step) / curriculum_den))
        else:
            curriculum_scale = 1.0

        selector_budget_weight = float(cfg.selector_budget_weight) * curriculum_scale
        selector_entropy_weight = float(cfg.selector_entropy_weight) * curriculum_scale

        total_loss_running = 0.0
        selector_ratio_running = 0.0
        selector_budget_running = 0.0
        selector_entropy_running = 0.0
        selector_metric_count = 0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)
            x0 = batch['motion'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            keyframes = batch['keyframes'].to(device, non_blocking=True)
            keyframe_indices = batch['keyframe_indices'].to(device, non_blocking=True)
            keyframe_mask = batch['keyframe_mask'].to(device, non_blocking=True)

            cond = _build_cond(batch, dataset, empty_emb, p_uncond=cfg.p_uncond)

            out = _compute_inbetween_loss(
                inbetween_model,
                diff_inbetween,
                x0,
                mask,
                keyframes,
                keyframe_indices,
                keyframe_mask,
                cond,
                cfg,
                keyframe_selector,
                selector_budget_weight=selector_budget_weight,
                selector_entropy_weight=selector_entropy_weight,
            )
            loss = out['loss']
            (loss / grad_accum_steps).backward()

            total_loss_running += loss.item()
            if out['selector_ratio'] is not None:
                selector_metric_count += 1
                selector_ratio_running += out['selector_ratio'].item()
                selector_budget_running += out['selector_budget'].item()
                selector_entropy_running += out['selector_entropy'].item()

        nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        inbetween_optimizer.step()

        total_loss_avg = total_loss_running / grad_accum_steps
        selector_ratio_avg = None
        selector_budget_avg = None
        selector_entropy_avg = None
        if selector_metric_count > 0:
            selector_ratio_avg = selector_ratio_running / selector_metric_count
            selector_budget_avg = selector_budget_running / selector_metric_count
            selector_entropy_avg = selector_entropy_running / selector_metric_count

        _update_ema_state(ema_inbetween_state, inbetween_model, cfg.ema_decay)
        if keyframe_selector is not None and ema_selector_state is not None:
            _update_ema_state(ema_selector_state, keyframe_selector, cfg.ema_decay)

        val_loss = None
        if val_loader is not None and step % cfg.val_interval == 0:
            eval_inbetween_orig = _clone_state_dict(inbetween_model)
            eval_selector_orig = _clone_state_dict(keyframe_selector) if keyframe_selector is not None else None
            if cfg.use_ema_for_sampling:
                inbetween_model.load_state_dict(ema_inbetween_state)
                if keyframe_selector is not None and ema_selector_state is not None:
                    keyframe_selector.load_state_dict(ema_selector_state)

            val_loss = _evaluate_inbetween(
                inbetween_model,
                keyframe_selector,
                diff_inbetween,
                val_loader,
                val_loader.dataset,
                empty_emb,
                cfg,
                device,
            )

            inbetween_model.load_state_dict(eval_inbetween_orig)
            if keyframe_selector is not None and eval_selector_orig is not None:
                keyframe_selector.load_state_dict(eval_selector_orig)

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'inbetween': inbetween_model.state_dict(),
                    'inbetween_ema': ema_inbetween_state,
                    'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
                    'selector_ema': ema_selector_state if keyframe_selector is not None else None,
                    'optimizer': inbetween_optimizer.state_dict(),
                    'step': step,
                    'best_step': best_step,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg.__dict__,
                }, inbetween_best_ckpt_path)
                print(f"Saved NEW BEST checkpoint at step {step} (val_loss={best_val_loss:.6f})")
        
        if step % 200 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            steps_done = step - start_step
            steps_remaining = cfg.inbetween_steps - step
            avg_time_per_step = elapsed / steps_done
            eta_seconds = steps_remaining * avg_time_per_step
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)

            lr_inbetween = inbetween_optimizer.param_groups[0]['lr']
            lr_selector = inbetween_optimizer.param_groups[1]['lr'] if len(inbetween_optimizer.param_groups) > 1 else None

            if selector_ratio_avg is None:
                print(
                    f"step {step:>7d} | loss {total_loss_avg:.5f} | "
                    f"lr {lr_inbetween:.2e} | "
                    f"val {val_loss if val_loss is not None else float('nan'):.5f} | "
                    f"ETA: {eta_hours}h {eta_minutes}m"
                )
            else:
                print(
                    f"step {step:>7d} | loss {total_loss_avg:.5f} | "
                    f"sel_ratio {selector_ratio_avg:.4f} | "
                    f"sel_budget {selector_budget_avg:.5f} | "
                    f"sel_entropy {selector_entropy_avg:.5f} | "
                    f"lr_d {lr_inbetween:.2e} | "
                    f"lr_s {lr_selector if lr_selector is not None else float('nan'):.2e} | "
                    f"val {val_loss if val_loss is not None else float('nan'):.5f} | "
                    f"ETA: {eta_hours}h {eta_minutes}m"
                )

            _append_metrics_row(metrics_logger, {
                'step': step,
                'total_loss': total_loss_avg,
                'val_loss': val_loss,
                'lr_inbetween': lr_inbetween,
                'lr_selector': lr_selector,
                'selector_budget_weight': selector_budget_weight,
                'selector_entropy_weight': selector_entropy_weight,
                'selector_ratio': selector_ratio_avg,
                'selector_budget': selector_budget_avg,
                'selector_entropy': selector_entropy_avg,
            })

        if step % 2_000 == 0:
            _save_convergence_plot(metrics_logger)
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'inbetween': inbetween_model.state_dict(),
                'inbetween_ema': ema_inbetween_state,
                'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
                'selector_ema': ema_selector_state if keyframe_selector is not None else None,
                'optimizer': inbetween_optimizer.state_dict(),
                'step': step,
                'best_step': best_step,
                'best_val_loss': best_val_loss,
                'cfg': cfg.__dict__,
            }, f'checkpoints/{ckpt_prefix}_step{step}.pt')
            print('Saved in-betweening checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'inbetween': inbetween_model.state_dict(),
        'inbetween_ema': ema_inbetween_state,
        'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
        'selector_ema': ema_selector_state if keyframe_selector is not None else None,
        'optimizer': inbetween_optimizer.state_dict(),
        'step': cfg.inbetween_steps,
        'best_step': best_step,
        'best_val_loss': best_val_loss,
        'cfg': cfg.__dict__,
    }, inbetween_final_ckpt_path)
    print(f'Saved final in-betweening checkpoint: {inbetween_final_ckpt_path}')
    if best_step > 0:
        print(f'Best checkpoint: {inbetween_best_ckpt_path} (step={best_step}, val_loss={best_val_loss:.6f})')
    _save_convergence_plot(metrics_logger)
    print(f"In-between convergence plot: {metrics_logger['plot_path']}")
    
    print("Diffusion in-betweening training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train composite motion models')
    parser.add_argument(
        '--stage',
        choices=['all', 'vqvae', 'gpt', 'inbetween'],
        default='all',
        help='Training stage to run (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if final checkpoints exist'
    )
    parser.add_argument('--vqvae-steps', type=int, default=None, help='Override VQ-VAE training steps')
    parser.add_argument('--gpt-steps', type=int, default=None, help='Override GPT training steps')
    parser.add_argument('--inbetween-steps', type=int, default=None, help='Override in-betweening training steps')
    parser.add_argument('--inbetween-resume', type=str, default=None, help='Path to in-betweening checkpoint to resume/fine-tune from')
    parser.add_argument('--inbetween-ckpt-prefix', type=str, default='composite_inbetween', help='Prefix for in-betweening checkpoint filenames')
    parser.add_argument('--keyframe-source-dir', type=str, default=None, help='Directory of external conditioning motions (id.npy) used for keyframes')
    parser.add_argument('--disable-selector', action='store_true', help='Disable learned keyframe selector and use dataset keyframes directly')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader workers')
    parser.add_argument('--scheduler-type', choices=['cosine', 'constant'], default=None, help='Learning rate scheduler type')
    parser.add_argument('--warmup-ratio', type=float, default=None, help='Warmup ratio for LR schedule')
    parser.add_argument('--min-lr-ratio', type=float, default=None, help='Minimum LR ratio for cosine decay')
    parser.add_argument('--inbetween-lr', type=float, default=None, help='Override in-between model LR')
    parser.add_argument('--selector-lr', type=float, default=None, help='Override selector LR')
    parser.add_argument('--ema-decay', type=float, default=None, help='EMA decay for in-between model/selector')
    parser.add_argument('--selector-curriculum-fraction', type=float, default=None, help='Fraction of total steps to ramp selector regularization')
    parser.add_argument('--val-interval', type=int, default=None, help='Validation interval for in-between training')
    parser.add_argument('--val-batches', type=int, default=None, help='Number of validation batches per eval pass')
    parser.add_argument('--grad-accum-steps', type=int, default=None, help='Gradient accumulation steps')
    args = parser.parse_args()
    main(
        args.stage,
        args.force,
        vqvae_steps=args.vqvae_steps,
        gpt_steps=args.gpt_steps,
        inbetween_steps=args.inbetween_steps,
        inbetween_resume=args.inbetween_resume,
        inbetween_ckpt_prefix=args.inbetween_ckpt_prefix,
        keyframe_source_dir=args.keyframe_source_dir,
        disable_selector=args.disable_selector,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        inbetween_lr=args.inbetween_lr,
        selector_lr=args.selector_lr,
        ema_decay=args.ema_decay,
        selector_curriculum_fraction=args.selector_curriculum_fraction,
        val_interval=args.val_interval,
        val_batches=args.val_batches,
        grad_accum_steps=args.grad_accum_steps,
    )
