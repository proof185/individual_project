"""Train the reconstruction keyframe selector against external CondMDI."""

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
from torch.utils.data import DataLoader

from config import ReconstructionTrainConfig
from condmdi_adapter import load_external_condmdi_runtime
from dataset import HUMANML3DCompositeDataset, collate_composite
from models.selector_modules import SELECTOR_MODE_CHOICES, build_keyframe_selector
from utils import encode_text, encode_text_with_tokens, setup_clip_model


class TextEncoderWrapper:
    """Wrapper for text encoder that can be pickled for multiprocessing."""

    def __init__(self, clip_model):
        self.clip_model = clip_model

    def __call__(self, texts, normalize=True):
        return encode_text(self.clip_model, texts, normalize)

    def encode_with_tokens(self, texts, normalize=True):
        return encode_text_with_tokens(self.clip_model, texts, normalize)


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
    if not logger['rows']:
        return

    import matplotlib.pyplot as plt

    numeric_cols = [c for c in logger['columns'] if c != 'step']
    if not numeric_cols:
        return

    series = {c: [] for c in numeric_cols}
    steps = []
    for row in logger['rows']:
        step = _safe_float(row.get('step'))
        if step is None:
            continue

        parsed = {}
        has_any = False
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


def _default_selector_ckpt_prefix(cfg: ReconstructionTrainConfig) -> str:
    mode = str(cfg.selector_mode).strip().lower().replace('-', '_')
    return f'composite_selector_{mode}'


def _batch_text_prompts(batch, dataset) -> list[str]:
    texts_by_id = {item['id']: item.get('texts', ['']) for item in dataset.data}
    prompts = []
    for sample_id, text_idx in zip(batch['ids'], batch['text_idxs']):
        options = texts_by_id.get(sample_id, [''])
        if not options:
            prompts.append('')
            continue
        prompts.append(options[min(int(text_idx), len(options) - 1)].split('#')[0].strip())
    return prompts


def _endpoint_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    endpoints = torch.zeros_like(valid_mask, dtype=torch.float32)
    has_valid = valid_mask.any(dim=1)
    if has_valid.any():
        batch_idx = torch.arange(valid_mask.shape[0], device=valid_mask.device)[has_valid]
        last_idx = valid_mask.long().sum(dim=1).clamp(min=1) - 1
        endpoints[batch_idx, 0] = 1.0
        endpoints[batch_idx, last_idx[batch_idx]] = 1.0
    return endpoints * valid_mask.float()


def _sample_observation_mask(probs: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    endpoints = _endpoint_mask(valid_mask)
    inner_mask = (valid_mask.float() * (1.0 - endpoints)).bool()
    p = probs.clamp(1e-5, 1.0 - 1e-5)
    sample = torch.bernoulli(p) * inner_mask.float()
    sample = torch.maximum(sample, endpoints) * valid_mask.float()

    log_prob = sample * torch.log(p) + (1.0 - sample) * torch.log(1.0 - p)
    log_prob = (log_prob * inner_mask.float()).sum(dim=1) / inner_mask.float().sum(dim=1).clamp(min=1.0)
    return sample, log_prob


@torch.no_grad()
def _condmdi_reconstruction_losses(
    diff_inbetween,
    x0: torch.Tensor,
    mask: torch.Tensor,
    observation_mask: torch.Tensor,
    text_prompts: list[str],
    cfg: ReconstructionTrainConfig,
) -> torch.Tensor:
    timestep_count = max(1, int(getattr(cfg, 'selector_reconstruction_timesteps', 3)))
    max_t = max(1, int(getattr(cfg, 'T_diffusion', 1000)) - 1)
    timesteps = torch.linspace(1, max_t, steps=timestep_count, device=x0.device).round().long().tolist()

    observed = observation_mask.bool() & mask.bool()
    weights = (mask.float() * (~observed).float()).clamp(min=0.0)
    if (weights.sum(dim=1) <= 0).any():
        weights = mask.float()

    losses = []
    for t in timesteps:
        t_batch = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
        x0_hat = diff_inbetween.predict_x0_local(
            x0,
            mask,
            t_batch,
            observed,
            text_prompts,
        )
        per_sample = ((x0 - x0_hat) ** 2 * weights.unsqueeze(-1)).sum(dim=(1, 2)) / (
            weights.sum(dim=1).clamp(min=1e-8) * x0.shape[-1]
        )
        losses.append(per_sample)
    return torch.stack(losses, dim=0).mean(dim=0)


def _reconstruction_selector_train_loss(
    keyframe_selector,
    diff_inbetween,
    x0: torch.Tensor,
    mask: torch.Tensor,
    cond: torch.Tensor,
    text_prompts: list[str],
    cfg: ReconstructionTrainConfig,
    selector_budget_weight: float,
    selector_entropy_weight: float,
):
    selector_probs, _ = keyframe_selector(x0, mask, cond=cond)
    sampled_mask, log_prob = _sample_observation_mask(selector_probs, mask)
    reconstruction_losses = _condmdi_reconstruction_losses(
        diff_inbetween,
        x0,
        mask,
        sampled_mask,
        text_prompts,
        cfg,
    )

    selector_ratio = (selector_probs * mask.float()).sum() / (mask.float().sum() + 1e-8)
    selector_budget_loss = (selector_ratio - cfg.selector_target_ratio) ** 2
    p = selector_probs.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    selector_entropy_loss = (entropy * mask.float()).sum() / (mask.float().sum() + 1e-8)

    advantage = reconstruction_losses.detach()
    if advantage.numel() > 1:
        advantage = advantage - advantage.mean()
    policy_loss = (advantage * log_prob).mean()
    loss = (
        float(cfg.selector_aux_weight) * policy_loss
        + selector_budget_weight * selector_budget_loss
        + selector_entropy_weight * selector_entropy_loss
    )
    return {
        'loss': loss,
        'selector_ratio': selector_ratio,
        'selector_budget': selector_budget_loss,
        'selector_entropy': selector_entropy_loss,
        'selector_aux': reconstruction_losses.detach().mean(),
    }


@torch.no_grad()
def _evaluate_reconstruction_selector(
    keyframe_selector,
    diff_inbetween,
    val_loader,
    dataset,
    empty_emb,
    cfg: ReconstructionTrainConfig,
    device: torch.device,
):
    keyframe_selector.eval()
    losses = []
    max_batches = max(1, int(cfg.val_batches))
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x0 = batch['motion'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        cond = _build_cond(batch, dataset, empty_emb, p_uncond=0.0)
        text_prompts = _batch_text_prompts(batch, dataset)
        _, selector_mask_st = keyframe_selector(x0, mask, cond=cond)
        batch_losses = _condmdi_reconstruction_losses(
            diff_inbetween,
            x0,
            mask,
            selector_mask_st > 0.5,
            text_prompts,
            cfg,
        )
        losses.extend(batch_losses.detach().cpu().tolist())
    keyframe_selector.train()
    if not losses:
        return None
    return float(sum(losses) / len(losses))


def main(
    force_retrain: bool = False,
    resume: str | None = None,
    condmdi_ckpt: str | None = None,
    device_arg: str | None = None,
):
    device = torch.device(device_arg or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('device:', device)

    cfg = ReconstructionTrainConfig()
    cfg.selector_mode = str(cfg.selector_mode).strip().lower()
    if cfg.selector_mode not in SELECTOR_MODE_CHOICES:
        raise ValueError(f"Unknown selector mode {cfg.selector_mode!r}. Expected one of {SELECTOR_MODE_CHOICES}.")
    if cfg.selector_mode != 'reconstruction':
        raise ValueError('train_reconstruction.py trains only the reconstruction selector. Use random/uniform/saliency as baselines.')
    if condmdi_ckpt:
        cfg.external_inbetween_ckpt_path = condmdi_ckpt

    selector_ckpt_prefix = _default_selector_ckpt_prefix(cfg)

    external_ckpt = os.path.abspath(cfg.external_inbetween_ckpt_path)
    
    if not os.path.exists(external_ckpt):
        raise FileNotFoundError(f'External CondMDI checkpoint not found: {external_ckpt}')

    print(f'Selector mode: {cfg.selector_mode}')
    print(f'External CondMDI checkpoint: {external_ckpt}')

    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    feature_dim = mean.shape[0]
    print('Feature dim:', feature_dim)

    _, diff_inbetween = load_external_condmdi_runtime(
        checkpoint_path=external_ckpt,
        local_mean=mean,
        local_std=std,
        device=str(device),
    )
    print('Loaded external CondMDI runtime for reconstruction rewards')

    clip_model = setup_clip_model(device)
    text_encoder = TextEncoderWrapper(clip_model)
    empty_emb, _, _ = text_encoder.encode_with_tokens([''])
    empty_emb = empty_emb.squeeze(0)
    print('Empty embedding norm:', empty_emb.norm().item())

    dataset = HUMANML3DCompositeDataset(
        cfg.root,
        split='train',
        max_len=cfg.max_len,
        normalize=True,
        use_cache=True,
        text_encoder=text_encoder,
        keyframe_interval=5,
        keyframe_strategy=cfg.keyframe_strategy,
        keyframe_count=None,
        keyframe_min=8,
        keyframe_max=28,
        keyframe_include_ends=True,
        include_keyframes=True,
        conditioning_motion_dir=None,
        mean=mean,
        std=std,
        device=device,
        load_token_embeddings=False,
    )
    print('Dataset size:', len(dataset))

    val_dataset = None
    val_loader = None
    val_split_file = os.path.join(cfg.root, f'{cfg.val_split}.txt')
    if os.path.exists(val_split_file):
        val_dataset = HUMANML3DCompositeDataset(
            cfg.root,
            split=cfg.val_split,
            max_len=cfg.max_len,
            normalize=True,
            use_cache=True,
            text_encoder=text_encoder,
            keyframe_interval=5,
            keyframe_strategy=cfg.keyframe_strategy,
            keyframe_count=None,
            keyframe_min=8,
            keyframe_max=28,
            keyframe_include_ends=True,
            conditioning_motion_dir=None,
            mean=mean,
            std=std,
            device=device,
            load_token_embeddings=False,
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
        print(f'Validation dataset size: {len(val_dataset)} ({cfg.val_split})')
    else:
        print(f'Validation split file not found: {val_split_file}; best-checkpoint selection disabled.')

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
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}",
    )

    keyframe_selector = build_keyframe_selector(
        mode=cfg.selector_mode,
        feature_dim=feature_dim,
        cond_dim=512,
        d_model=cfg.selector_d_model,
        n_layers=cfg.selector_layers,
        n_heads=cfg.selector_heads,
        dropout=cfg.selector_dropout,
        max_len=cfg.max_len + 10,
        threshold=cfg.selector_threshold,
        budget_ratio=cfg.selector_target_ratio,
    ).to(device)
    print(
        f"Keyframe selector [{cfg.selector_mode}] params: "
        f"{sum(p.numel() for p in keyframe_selector.parameters())/1e6:.2f}M"
    )

    selector_final_ckpt_path = f'checkpoints/{selector_ckpt_prefix}_step{cfg.selector_steps}.pt'
    selector_best_ckpt_path = f'checkpoints/{selector_ckpt_prefix}_best.pt'
    if os.path.exists(selector_final_ckpt_path) and not force_retrain and not resume:
        print(f'Selector training already complete (found final checkpoint): {selector_final_ckpt_path}')
        return

    selector_lr = float(cfg.selector_lr)
    selector_params = [p for p in keyframe_selector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': selector_params, 'lr': selector_lr, 'name': 'selector'}])
    base_lrs = [selector_lr]
    ema_selector_state = _clone_state_dict(keyframe_selector)

    metrics_logger = _init_metrics_logger(
        'selector',
        ['step', 'total_loss', 'val_loss', 'selector_ratio', 'selector_aux'],
    )
    print(f'Selector metrics CSV: {metrics_logger["csv_path"]}')
    print(f'Stage: selector-only training against external CondMDI objectives ({cfg.selector_mode})')

    start_step = 0
    best_val_loss = float('inf')
    best_step = 0
    if resume and os.path.exists(resume) and not force_retrain:
        print(f'Resuming selector from checkpoint: {resume}')
        ckpt = torch.load(resume, map_location=device)
        if ckpt.get('selector') is not None:
            keyframe_selector.load_state_dict(ckpt['selector'])
        if ckpt.get('selector_ema') is not None:
            ema_selector_state = {k: v.detach().clone() for k, v in ckpt['selector_ema'].items()}
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_step = int(ckpt.get('step', 0))
        print(f'Resume step: {start_step}')

    keyframe_selector.train()
    data_iter = cycle(loader)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    start_time = time.time()

    for step in range(start_step + 1, cfg.selector_steps + 1):
        _apply_lr_schedule(
            optimizer,
            base_lrs,
            step,
            cfg.selector_steps,
            cfg.warmup_ratio,
            cfg.min_lr_ratio,
            cfg.scheduler_type,
        )
        optimizer.zero_grad(set_to_none=True)

        curriculum_den = max(1.0, float(cfg.selector_steps) * float(cfg.selector_curriculum_fraction))
        curriculum_scale = min(1.0, max(0.0, float(step) / curriculum_den))
        selector_budget_weight = float(cfg.selector_budget_weight) * curriculum_scale
        selector_entropy_weight = float(cfg.selector_entropy_weight) * curriculum_scale

        total_loss_running = 0.0
        selector_ratio_running = 0.0
        selector_aux_running = 0.0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)
            x0 = batch['motion'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            cond = _build_cond(batch, dataset, empty_emb, p_uncond=cfg.p_uncond)
            text_prompts = _batch_text_prompts(batch, dataset)

            out = _reconstruction_selector_train_loss(
                keyframe_selector,
                diff_inbetween,
                x0,
                mask,
                cond,
                text_prompts,
                cfg,
                selector_budget_weight,
                selector_entropy_weight,
            )
            loss = out['loss']
            (loss / grad_accum_steps).backward()

            total_loss_running += loss.item()
            selector_ratio_running += out['selector_ratio'].item()
            selector_aux_running += out['selector_aux'].item()

        nn.utils.clip_grad_norm_(selector_params, cfg.grad_clip)
        optimizer.step()
        _update_ema_state(ema_selector_state, keyframe_selector, cfg.ema_decay)

        total_loss_avg = total_loss_running / grad_accum_steps
        selector_ratio_avg = selector_ratio_running / grad_accum_steps
        selector_aux_avg = selector_aux_running / grad_accum_steps

        val_loss = None
        if val_loader is not None and step % cfg.val_interval == 0:
            eval_selector_orig = _clone_state_dict(keyframe_selector)
            if cfg.use_ema_for_sampling and ema_selector_state is not None:
                keyframe_selector.load_state_dict(ema_selector_state)

            val_loss = _evaluate_reconstruction_selector(
                keyframe_selector,
                diff_inbetween,
                val_loader,
                val_loader.dataset,
                empty_emb,
                cfg,
                device,
            )
            keyframe_selector.load_state_dict(eval_selector_orig)

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'inbetween': None,
                    'inbetween_ema': None,
                    'selector': keyframe_selector.state_dict(),
                    'selector_ema': ema_selector_state,
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'best_step': best_step,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg.__dict__,
                }, selector_best_ckpt_path)
                print(f'Saved NEW BEST selector checkpoint at step {step} (val_loss={best_val_loss:.6f})')

        if step % 200 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            steps_done = step - start_step
            steps_remaining = cfg.selector_steps - step
            avg_time_per_step = elapsed / max(1, steps_done)
            eta_seconds = steps_remaining * avg_time_per_step
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            print(
                f"step {step:>7d} | loss {total_loss_avg:.5f} | "
                f"sel_ratio {selector_ratio_avg:.4f} | "
                f"sel_aux {selector_aux_avg:.5f} | "
                f"val {val_loss if val_loss is not None else float('nan'):.5f} | "
                f"ETA: {eta_hours}h {eta_minutes}m"
            )
            _append_metrics_row(metrics_logger, {
                'step': step,
                'total_loss': total_loss_avg,
                'val_loss': val_loss,
                'selector_ratio': selector_ratio_avg,
                'selector_aux': selector_aux_avg,
            })

        if step % 2_000 == 0:
            _save_convergence_plot(metrics_logger)

        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'inbetween': None,
                'inbetween_ema': None,
                'selector': keyframe_selector.state_dict(),
                'selector_ema': ema_selector_state,
                'optimizer': optimizer.state_dict(),
                'step': step,
                'best_step': best_step,
                'best_val_loss': best_val_loss,
                'cfg': cfg.__dict__,
            }, f'checkpoints/{selector_ckpt_prefix}_step{step}.pt')
            print('Saved selector checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'inbetween': None,
        'inbetween_ema': None,
        'selector': keyframe_selector.state_dict(),
        'selector_ema': ema_selector_state,
        'optimizer': optimizer.state_dict(),
        'step': cfg.selector_steps,
        'best_step': best_step,
        'best_val_loss': best_val_loss,
        'cfg': cfg.__dict__,
    }, selector_final_ckpt_path)
    print(f'Saved final selector checkpoint: {selector_final_ckpt_path}')
    if best_step > 0:
        print(f'Best checkpoint: {selector_best_ckpt_path} (step={best_step}, val_loss={best_val_loss:.6f})')
    _save_convergence_plot(metrics_logger)
    print(f'Selector convergence plot: {metrics_logger["plot_path"]}')
    print('Selector training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the reconstruction keyframe selector against external CondMDI')
    parser.add_argument('--force', action='store_true', help='Force retraining even if the final checkpoint exists')
    parser.add_argument('--resume', type=str, default=None, help='Path to reconstruction selector checkpoint to resume from')
    parser.add_argument('--condmdi-ckpt', type=str, default=None, help='Path to the external CondMDI checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Training device override, e.g. cuda, cuda:0, or cpu')
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(
            'Warning: ignoring non-essential CLI args. '
            'Set training hyperparameters in config.py instead.\n'
            f'Ignored args: {" ".join(unknown_args)}'
        )
    main(
        force_retrain=args.force,
        resume=args.resume,
        condmdi_ckpt=args.condmdi_ckpt,
        device_arg=args.device,
    )
