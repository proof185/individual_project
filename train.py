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
from models.diffusion import InbetweenDiffusion, InbetweenTransformer
from utils import masked_mse, vel_loss, setup_clip_model, encode_text, prepare_gpt_batch


def main(stage: str, force_retrain: bool):
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
    
    diff_inbetween = InbetweenDiffusion(cfg.T_diffusion, device=device)
    
    # Stage 1a: Train VQ-VAE
    if stage in {'all', 'vqvae'}:
        train_vqvae(vqvae, loader, cfg, device, mean, std, force_retrain)
    
    # Stage 1b: Train GPT
    if stage in {'all', 'gpt'}:
        train_gpt(vqvae, gpt, loader, dataset, empty_emb, cfg, device, force_retrain)
    
    # Stage 2: Train Diffusion In-betweening
    if stage in {'all', 'inbetween'}:
        train_inbetween(inbetween_model, diff_inbetween, loader, dataset, empty_emb, cfg, device, mean, std, force_retrain)
    
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
    
    print("MotionGPT training complete!")


def train_inbetween(inbetween_model, diff_inbetween, loader, dataset, empty_emb, cfg, device, mean, std, force_retrain: bool = False):
    """Stage 2: Train Diffusion In-betweening."""
    inbetween_final_ckpt_path = f'checkpoints/composite_inbetween_step{cfg.inbetween_steps}.pt'
    if os.path.exists(inbetween_final_ckpt_path) and not force_retrain:
        print(f"Loading In-Betweening model from final checkpoint: {inbetween_final_ckpt_path}")
        inbetween_ckpt = torch.load(inbetween_final_ckpt_path, map_location=device)
        inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
        print("Diffusion in-betweening training already complete! Loaded from checkpoint.")
        return
    
    inbetween_optimizer = torch.optim.AdamW(inbetween_model.parameters(), lr=cfg.lr)
    
    inbetween_model.train()
    data_iter = cycle(loader)
    
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
        
        # Replace keyframe positions with clean values (they're given)
        xt = diff_inbetween._replace_keyframes(xt, keyframes, keyframe_indices, keyframe_mask)
        
        # Predict x0
        x0_hat = inbetween_model(xt, t, cond, mask, keyframes, keyframe_indices, keyframe_mask)
        
        # Loss on non-keyframe positions (keyframes are given, no need to predict)
        non_keyframe_mask = mask.clone()
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            non_keyframe_mask[b, valid_idx] = False
        
        loss = masked_mse(x0, x0_hat, non_keyframe_mask)
        
        # Also add small loss on keyframes to ensure consistency
        keyframe_loss = masked_mse(x0, x0_hat, mask) * 0.1
        loss = loss + keyframe_loss
        
        # Optional: velocity smoothness
        v_loss_val = vel_loss(x0, x0_hat, mask) * 0.5
        loss = loss + v_loss_val
        
        inbetween_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(inbetween_model.parameters(), cfg.grad_clip)
        inbetween_optimizer.step()
        
        if step % 200 == 0:
            print(f"step {step:>7d} | loss {loss.item():.5f}")
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'inbetween': inbetween_model.state_dict(),
                'optimizer': inbetween_optimizer.state_dict(),
                'step': step,
                'cfg': cfg.__dict__,
            }, f'checkpoints/composite_inbetween_step{step}.pt')
            print('Saved in-betweening checkpoint')
    
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
    args = parser.parse_args()
    main(args.stage, args.force)
