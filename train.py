"""Training script for composite motion generation."""

import argparse
import os
import random
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CompositeConfig
from dataset import HUMANML3DCompositeDataset, collate_composite
from models.vqvae import MotionVQVAE
from models.gpt import MotionGPT
from models.diffusion import InbetweenDiffusion, InbetweenTransformer, KeyframeSelector
from utils import (
    boundary_acceleration_loss,
    boundary_velocity_loss,
    encode_text,
    masked_mse,
    prepare_gpt_batch,
    setup_clip_model,
    transition_consistency_loss,
    vel_loss,
    weighted_masked_mse,
)


def main(
    stage: str,
    force_retrain: bool,
    vqvae_steps: int | None = None,
    gpt_steps: int | None = None,
    inbetween_steps: int | None = None,
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
    print(f"Keyframe strategy: {cfg.keyframe_strategy}")
    
    # Load normalization statistics
    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    Fdim = mean.shape[0]
    print('Feature dim:', Fdim)
    
    # Setup CLIP
    clip_model = setup_clip_model(device)
    
    def text_encoder(texts, normalize=True):
        return encode_text(clip_model, texts, normalize)
    
    empty_emb = text_encoder([''])
    empty_emb = empty_emb.squeeze(0)
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
        mean=mean,
        std=std,
        device=device,
    )
    print('Dataset size:', len(dataset))
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_composite,
        drop_last=True
    )
    
    # Initialize models
    vqvae = MotionVQVAE(
        feature_dim=Fdim,
        codebook_size=cfg.codebook_size,
        codebook_dim=cfg.codebook_dim,
        downsample_rate=cfg.downsample_rate,
        commitment_cost=cfg.commitment_cost,
    ).to(device)
    print(f'VQ-VAE params: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M')
    
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
    
    inbetween_model = InbetweenTransformer(
        feature_dim=Fdim,
        cond_dim=512,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
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
        train_vqvae(vqvae, loader, cfg, device, mean, std, force_retrain)
    
    # Stage 1b: Train GPT
    if stage in {'all', 'gpt'}:
        train_gpt(vqvae, gpt, loader, dataset, empty_emb, cfg, device, force_retrain)
    
    # Stage 2: Train Diffusion In-betweening
    if stage in {'all', 'inbetween'}:
        train_inbetween(
            inbetween_model,
            diff_inbetween,
            loader,
            dataset,
            empty_emb,
            cfg,
            device,
            mean,
            std,
            force_retrain,
            keyframe_selector=keyframe_selector,
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
    lambda_recon = 1.0
    lambda_vel = 0.5
    
    vqvae.train()
    data_iter = cycle(loader)
    
    print("Stage 1a: Training VQ-VAE...")
    
    for step in range(1, cfg.vqvae_steps + 1):
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
        
        vqvae_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(vqvae.parameters(), cfg.grad_clip)
        vqvae_optimizer.step()
        
        if step % 200 == 0:
            print(f"step {step:>7d} | recon {recon_loss.item():.5f} | vq {vq_loss.item():.5f} | vel {v_loss.item():.5f}")
        
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
    
    print("VQ-VAE training complete!")


def train_gpt(vqvae, gpt, loader, dataset, empty_emb, cfg, device, force_retrain: bool = False):
    """Stage 1b: Train MotionGPT."""
    gpt_final_ckpt_path = f'checkpoints/composite_gpt_step{cfg.gpt_steps}.pt'
    if os.path.exists(gpt_final_ckpt_path) and not force_retrain:
        print(f"Loading GPT from final checkpoint: {gpt_final_ckpt_path}")
        gpt_ckpt = torch.load(gpt_final_ckpt_path, map_location=device)
        gpt.load_state_dict(gpt_ckpt['gpt'])
        vqvae.load_state_dict(gpt_ckpt['vqvae'])
        print("MotionGPT training already complete! Loaded from checkpoint.")
        return
    
    vqvae.eval()
    gpt_optimizer = torch.optim.AdamW(gpt.parameters(), lr=cfg.lr)
    
    gpt.train()
    data_iter = cycle(loader)
    
    print("Stage 1b: Training MotionGPT...")
    
    for step in range(1, cfg.gpt_steps + 1):
        batch = next(data_iter)
        
        input_tokens, target_tokens, target_mask = prepare_gpt_batch(batch, vqvae, gpt, cfg.downsample_rate, device)
        
        cond_list = []
        for mid, tidx in zip(batch['ids'], batch['text_idxs']):
            if random.random() < cfg.p_uncond:
                cond_list.append(empty_emb)
            else:
                cond_list.append(dataset.get_embedding(mid, tidx))
        cond = torch.stack(cond_list)
        
        logits = gpt(input_tokens, cond)
        
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            target_tokens.reshape(-1),
            ignore_index=gpt.pad_token,
            reduction='none'
        ).reshape(B, T)
        
        loss = (loss * target_mask.float()).sum() / (target_mask.float().sum() + 1e-8)
        
        gpt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(gpt.parameters(), cfg.grad_clip)
        gpt_optimizer.step()
        
        if step % 200 == 0:
            ppl = torch.exp(loss).item()
            print(f"step {step:>7d} | loss {loss.item():.5f} | ppl {ppl:.2f}")
        
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
    
    print("MotionGPT training complete!")


def _frame_mask_to_sparse_keyframes(
    x0: torch.Tensor,
    frame_mask: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """Convert per-frame keyframe mask (B, T) into padded sparse keyframe tensors."""
    B, T, F = x0.shape
    hard_mask = (frame_mask > 0.5) & valid_mask

    key_idx_list = []
    keyframe_list = []
    for b in range(B):
        idx = torch.nonzero(hard_mask[b], as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            idx = torch.tensor([0], device=x0.device, dtype=torch.long)
        key_idx_list.append(idx)
        keyframe_list.append(x0[b, idx])

    K_max = max(idx.numel() for idx in key_idx_list)
    keyframes = torch.zeros(B, K_max, F, device=x0.device, dtype=x0.dtype)
    keyframe_indices = torch.zeros(B, K_max, device=x0.device, dtype=torch.long)
    keyframe_mask = torch.zeros(B, K_max, device=x0.device, dtype=torch.bool)

    for b, (idx, kf) in enumerate(zip(key_idx_list, keyframe_list)):
        K = idx.numel()
        keyframes[b, :K] = kf
        keyframe_indices[b, :K] = idx
        keyframe_mask[b, :K] = True

    return keyframes, keyframe_indices, keyframe_mask


def train_inbetween(
    inbetween_model,
    diff_inbetween,
    loader,
    dataset,
    empty_emb,
    cfg,
    device,
    mean,
    std,
    force_retrain: bool = False,
    keyframe_selector=None,
):
    """Stage 2: Train Diffusion In-betweening."""
    inbetween_final_ckpt_path = f'checkpoints/composite_inbetween_step{cfg.inbetween_steps}.pt'
    if os.path.exists(inbetween_final_ckpt_path) and not force_retrain:
        print(f"Loading In-Betweening model from final checkpoint: {inbetween_final_ckpt_path}")
        inbetween_ckpt = torch.load(inbetween_final_ckpt_path, map_location=device)
        inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
        if keyframe_selector is not None and 'selector' in inbetween_ckpt:
            keyframe_selector.load_state_dict(inbetween_ckpt['selector'])
        print("Diffusion in-betweening training already complete! Loaded from checkpoint.")
        return
    
    params = list(inbetween_model.parameters())
    if keyframe_selector is not None:
        params += list(keyframe_selector.parameters())
    inbetween_optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    
    inbetween_model.train()
    if keyframe_selector is not None:
        keyframe_selector.train()
    data_iter = cycle(loader)
    
    if keyframe_selector is not None:
        print(f"Stage 2: Training Diffusion In-Betweening with LEARNED keyframe selector (target ratio={cfg.selector_target_ratio})...")
    else:
        print(f"Stage 2: Training Diffusion In-Betweening (keyframe interval={cfg.keyframe_interval})...")
    
    for step in range(1, cfg.inbetween_steps + 1):
        batch = next(data_iter)
        x0 = batch['motion'].to(device)
        mask = batch['mask'].to(device)
        keyframes = batch['keyframes'].to(device)
        keyframe_indices = batch['keyframe_indices'].to(device)
        keyframe_mask = batch['keyframe_mask'].to(device)
        
        # Get text condition (with CFG dropout)
        cond_list = []
        for mid, tidx in zip(batch['ids'], batch['text_idxs']):
            if random.random() < cfg.p_uncond:
                cond_list.append(empty_emb)
            else:
                cond_list.append(dataset.get_embedding(mid, tidx))
        cond = torch.stack(cond_list)
        
        # Sample diffusion timestep
        B = x0.shape[0]
        t = torch.randint(0, cfg.T_diffusion, (B,), device=device)
        
        # Add noise
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

            # Train diffusion to reconstruct mostly where selector did not observe.
            non_keyframe_weights = (1.0 - selector_mask_st) * mask.float()
            loss = weighted_masked_mse(x0, x0_hat, non_keyframe_weights, mask)

            selector_ratio = (selector_probs * mask.float()).sum() / (mask.float().sum() + 1e-8)
            selector_budget_loss = (selector_ratio - cfg.selector_target_ratio) ** 2
            p = selector_probs.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            selector_entropy_loss = (entropy * mask.float()).sum() / (mask.float().sum() + 1e-8)
            loss = (
                loss
                + cfg.selector_budget_weight * selector_budget_loss
                + cfg.selector_entropy_weight * selector_entropy_loss
            )

            # Boundary / transition losses use hard keyframe indices for locality.
            keyframes, keyframe_indices, keyframe_mask = _frame_mask_to_sparse_keyframes(
                x0,
                selector_mask_st.detach(),
                mask,
            )
        else:
            # Replace keyframe positions with clean values (they're given).
            xt = diff_inbetween._replace_keyframes(xt, keyframes, keyframe_indices, keyframe_mask)

            # Predict x0.
            x0_hat = inbetween_model(xt, t, cond, mask, keyframes, keyframe_indices, keyframe_mask)

            # Loss on non-keyframe positions (keyframes are given, no need to predict).
            non_keyframe_mask = mask.clone()
            for b in range(B):
                valid = keyframe_mask[b]
                valid_idx = keyframe_indices[b][valid]
                non_keyframe_mask[b, valid_idx] = False
            loss = masked_mse(x0, x0_hat, non_keyframe_mask)

        # Encourage smooth entry/exit around keyframes without forcing the
        # entire segment into a linear interpolation that flattens jumps.
        boundary_vel = boundary_velocity_loss(x0, x0_hat, mask, keyframe_indices, keyframe_mask)
        loss = loss + cfg.boundary_velocity_weight * boundary_vel

        boundary_accel = boundary_acceleration_loss(x0, x0_hat, mask, keyframe_indices, keyframe_mask)
        loss = loss + cfg.boundary_acceleration_weight * boundary_accel

        # Keep a light global velocity term to avoid drift, but do not over-smooth.
        v_loss_val = vel_loss(x0, x0_hat, mask) * cfg.inbetween_velocity_weight
        loss = loss + v_loss_val

        # Replace global jerk regularization with keyframe-local transition matching.
        transition_loss_val = transition_consistency_loss(
            x0,
            x0_hat,
            mask,
            keyframe_indices,
            keyframe_mask,
            window=cfg.transition_window,
            sigma=cfg.transition_sigma,
            velocity_weight=cfg.transition_velocity_weight,
            acceleration_weight=cfg.transition_acceleration_weight,
        )
        loss = loss + cfg.transition_consistency_weight * transition_loss_val
        
        inbetween_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(inbetween_model.parameters(), cfg.grad_clip)
        inbetween_optimizer.step()
        
        if step % 200 == 0:
            if selector_ratio is None:
                print(f"step {step:>7d} | loss {loss.item():.5f}")
            else:
                print(
                    f"step {step:>7d} | loss {loss.item():.5f} | "
                    f"sel_ratio {selector_ratio.item():.4f} | "
                    f"sel_budget {selector_budget_loss.item():.5f} | "
                    f"sel_entropy {selector_entropy_loss.item():.5f}"
                )
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'inbetween': inbetween_model.state_dict(),
                'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
                'optimizer': inbetween_optimizer.state_dict(),
                'step': step,
                'cfg': cfg.__dict__,
            }, f'checkpoints/composite_inbetween_step{step}.pt')
            print('Saved in-betweening checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'inbetween': inbetween_model.state_dict(),
        'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
        'optimizer': inbetween_optimizer.state_dict(),
        'step': cfg.inbetween_steps,
        'cfg': cfg.__dict__,
    }, inbetween_final_ckpt_path)
    print(f'Saved final in-betweening checkpoint: {inbetween_final_ckpt_path}')
    
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
    args = parser.parse_args()
    main(
        args.stage,
        args.force,
        vqvae_steps=args.vqvae_steps,
        gpt_steps=args.gpt_steps,
        inbetween_steps=args.inbetween_steps,
    )
