"""Selector-only training script backed by external CondMDI."""

import argparse
import os
import time
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import SelectorTrainConfig
from dataset import HUMANML3DCompositeDataset, collate_composite
from models.selectors import SELECTOR_MODE_CHOICES, build_keyframe_selector
from train import (
    TextEncoderWrapper,
    _append_metrics_row,
    _apply_lr_schedule,
    _build_cond,
    _clone_state_dict,
    _default_selector_eval_root,
    _evaluate_inbetween,
    _gather_selector_oracle_targets,
    _init_metrics_logger,
    _prepare_selector_oracle_targets,
    _save_convergence_plot,
    _selector_mode_uses_oracle_targets,
    _update_ema_state,
)
from utils import encode_text_with_tokens, setup_clip_model


def _default_selector_ckpt_prefix(cfg: SelectorTrainConfig) -> str:
    mode = str(cfg.selector_mode).strip().lower().replace('-', '_')
    return f'composite_selector_{mode}'


def main(force_retrain: bool = False, selector_resume: str | None = None, selector_ckpt_prefix: str | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    cfg = SelectorTrainConfig()
    cfg.selector_mode = str(cfg.selector_mode).strip().lower()
    if cfg.selector_mode not in SELECTOR_MODE_CHOICES:
        raise ValueError(f"Unknown selector mode {cfg.selector_mode!r}. Expected one of {SELECTOR_MODE_CHOICES}.")

    if selector_ckpt_prefix is None or not selector_ckpt_prefix.strip():
        selector_ckpt_prefix = _default_selector_ckpt_prefix(cfg)
    selector_ckpt_prefix = selector_ckpt_prefix.strip()

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

    selector_oracle_targets = None
    val_selector_oracle_targets = None
    if _selector_mode_uses_oracle_targets(cfg.selector_mode):
        oracle_ckpt_path = cfg.selector_oracle_ckpt_path or external_ckpt
        oracle_ckpt_path = os.path.abspath(oracle_ckpt_path)
        selector_oracle_targets = _prepare_selector_oracle_targets(dataset, cfg, device, oracle_ckpt_path)
        if val_dataset is not None:
            val_selector_oracle_targets = _prepare_selector_oracle_targets(val_dataset, cfg, device, oracle_ckpt_path)

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
    print('Optimization target: selector parameters only. External CondMDI is fixed.')

    selector_final_ckpt_path = f'checkpoints/{selector_ckpt_prefix}_step{cfg.selector_steps}.pt'
    selector_best_ckpt_path = f'checkpoints/{selector_ckpt_prefix}_best.pt'
    if os.path.exists(selector_final_ckpt_path) and not force_retrain and not selector_resume:
        print(f'Loading selector from final checkpoint: {selector_final_ckpt_path}')
        ckpt = torch.load(selector_final_ckpt_path, map_location=device)
        selector_key = 'selector_ema' if cfg.use_ema_for_sampling and ckpt.get('selector_ema') is not None else 'selector'
        if ckpt.get(selector_key) is not None:
            keyframe_selector.load_state_dict(ckpt[selector_key])
        print('Selector training already complete! Loaded from checkpoint.')
        return

    selector_lr = float(cfg.selector_lr)
    selector_params = [p for p in keyframe_selector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': selector_params, 'lr': selector_lr, 'name': 'selector'}])
    base_lrs = [selector_lr]
    ema_selector_state = _clone_state_dict(keyframe_selector)

    metrics_logger = _init_metrics_logger(
        'selector',
        [
            'step', 'total_loss', 'val_loss', 'lr_selector',
            'selector_budget_weight', 'selector_entropy_weight', 'selector_aux_weight',
            'selector_ratio', 'selector_budget', 'selector_entropy', 'selector_aux',
        ],
    )
    print(f'Selector metrics CSV: {metrics_logger["csv_path"]}')
    print(f'Stage: selector-only training against external CondMDI objectives ({cfg.selector_mode})')

    start_step = 0
    best_val_loss = float('inf')
    best_step = 0
    if selector_resume and os.path.exists(selector_resume) and not force_retrain:
        print(f'Resuming selector from checkpoint: {selector_resume}')
        ckpt = torch.load(selector_resume, map_location=device)
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
        selector_budget_running = 0.0
        selector_entropy_running = 0.0
        selector_aux_running = 0.0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)
            x0 = batch['motion'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            cond = _build_cond(batch, dataset, empty_emb, p_uncond=cfg.p_uncond)
            selector_oracle_target = _gather_selector_oracle_targets(batch, selector_oracle_targets, device)

            selector_probs, selector_mask_st = keyframe_selector(x0, mask, cond=cond)
            selector_ratio = (selector_probs * mask.float()).sum() / (mask.float().sum() + 1e-8)
            selector_budget_loss = (selector_ratio - cfg.selector_target_ratio) ** 2
            p = selector_probs.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            selector_entropy_loss = (entropy * mask.float()).sum() / (mask.float().sum() + 1e-8)
            selector_aux_loss = keyframe_selector.compute_auxiliary_loss(
                x0,
                mask,
                cond,
                selector_probs,
                selector_mask_st,
                oracle_target=selector_oracle_target,
            )
            loss = (
                selector_budget_weight * selector_budget_loss
                + selector_entropy_weight * selector_entropy_loss
                + float(cfg.selector_aux_weight) * selector_aux_loss
            )
            (loss / grad_accum_steps).backward()

            total_loss_running += loss.item()
            selector_ratio_running += selector_ratio.item()
            selector_budget_running += selector_budget_loss.item()
            selector_entropy_running += selector_entropy_loss.item()
            selector_aux_running += selector_aux_loss.item()

        nn.utils.clip_grad_norm_(selector_params, cfg.grad_clip)
        optimizer.step()
        _update_ema_state(ema_selector_state, keyframe_selector, cfg.ema_decay)

        total_loss_avg = total_loss_running / grad_accum_steps
        selector_ratio_avg = selector_ratio_running / grad_accum_steps
        selector_budget_avg = selector_budget_running / grad_accum_steps
        selector_entropy_avg = selector_entropy_running / grad_accum_steps
        selector_aux_avg = selector_aux_running / grad_accum_steps

        val_loss = None
        if val_loader is not None and step % cfg.val_interval == 0:
            eval_selector_orig = _clone_state_dict(keyframe_selector)
            if cfg.use_ema_for_sampling and ema_selector_state is not None:
                keyframe_selector.load_state_dict(ema_selector_state)

            val_loss = _evaluate_inbetween(
                None,
                keyframe_selector,
                None,
                val_loader,
                val_loader.dataset,
                empty_emb,
                cfg,
                device,
                selector_oracle_targets=val_selector_oracle_targets,
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
            lr_selector = optimizer.param_groups[0]['lr']
            print(
                f"step {step:>7d} | loss {total_loss_avg:.5f} | "
                f"sel_ratio {selector_ratio_avg:.4f} | "
                f"sel_budget {selector_budget_avg:.5f} | "
                f"sel_entropy {selector_entropy_avg:.5f} | "
                f"sel_aux {selector_aux_avg:.5f} | "
                f"lr_selector {lr_selector:.2e} | "
                f"val {val_loss if val_loss is not None else float('nan'):.5f} | "
                f"ETA: {eta_hours}h {eta_minutes}m"
            )
            _append_metrics_row(metrics_logger, {
                'step': step,
                'total_loss': total_loss_avg,
                'val_loss': val_loss,
                'lr_selector': lr_selector,
                'selector_budget_weight': selector_budget_weight,
                'selector_entropy_weight': selector_entropy_weight,
                'selector_aux_weight': cfg.selector_aux_weight,
                'selector_ratio': selector_ratio_avg,
                'selector_budget': selector_budget_avg,
                'selector_entropy': selector_entropy_avg,
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
    parser = argparse.ArgumentParser(description='Train keyframe selectors against external CondMDI')
    parser.add_argument('--force', action='store_true', help='Force retraining even if the final checkpoint exists')
    parser.add_argument('--selector-resume', type=str, default=None, help='Path to selector checkpoint to resume from')
    parser.add_argument('--selector-ckpt-prefix', type=str, default=None, help='Prefix for selector checkpoint filenames')
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(
            'Warning: ignoring non-essential CLI args. '
            'Set training hyperparameters in config.py instead.\n'
            f'Ignored args: {" ".join(unknown_args)}'
        )
    main(
        force_retrain=args.force,
        selector_resume=args.selector_resume,
        selector_ckpt_prefix=args.selector_ckpt_prefix,
    )
