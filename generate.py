"""Inference functions for composite motion generation."""

import os
import random

import numpy as np
import torch

from config import CompositeConfig
from models.vqvae import MotionVQVAE
from models.gpt import MotionGPT
from models.diffusion import InbetweenDiffusion, InbetweenTransformer
from utils import setup_clip_model, encode_text


def load_models(cfg: CompositeConfig, device: str = 'cuda'):
    """Load trained models from checkpoints."""
    # Load normalization statistics
    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    Fdim = mean.shape[0]
    
    # Initialize models
    vqvae = MotionVQVAE(
        feature_dim=Fdim,
        codebook_size=cfg.codebook_size,
        codebook_dim=cfg.codebook_dim,
        downsample_rate=cfg.downsample_rate,
        commitment_cost=cfg.commitment_cost,
    ).to(device)
    
    gpt = MotionGPT(
        codebook_size=cfg.codebook_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        max_len=cfg.max_len // cfg.downsample_rate + 10,
        cond_dim=512,
    ).to(device)
    
    inbetween_model = InbetweenTransformer(
        feature_dim=Fdim,
        cond_dim=512,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        max_len=cfg.max_len + 10,
    ).to(device)
    
    diff_inbetween = InbetweenDiffusion(cfg.T_diffusion, device=device)
    
    # Load checkpoints
    gpt_ckpt_path = f'checkpoints/composite_gpt_step{cfg.gpt_steps}.pt'
    if os.path.exists(gpt_ckpt_path):
        gpt_ckpt = torch.load(gpt_ckpt_path, map_location=device)
        gpt.load_state_dict(gpt_ckpt['gpt'])
        vqvae.load_state_dict(gpt_ckpt['vqvae'])
        print(f"Loaded GPT and VQ-VAE from {gpt_ckpt_path}")
    
    inbetween_ckpt_path = f'checkpoints/composite_inbetween_step{cfg.inbetween_steps}.pt'
    if os.path.exists(inbetween_ckpt_path):
        inbetween_ckpt = torch.load(inbetween_ckpt_path, map_location=device)
        inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
        print(f"Loaded in-betweening model from {inbetween_ckpt_path}")
    
    # Set to eval mode
    vqvae.eval()
    gpt.eval()
    inbetween_model.eval()
    
    # Setup CLIP
    clip_model = setup_clip_model(device)
    
    return vqvae, gpt, inbetween_model, diff_inbetween, clip_model, mean, std, Fdim


def _select_keyframe_indices(
    length: int,
    keyframe_interval: int,
    strategy: str,
    keyframe_count: int | None,
    keyframe_min: int,
    keyframe_max: int,
    include_ends: bool,
):
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


@torch.no_grad()
def generate_composite(
    vqvae,
    gpt,
    inbetween_model,
    diff_inbetween,
    clip_model,
    mean,
    std,
    Fdim,
    cfg,
    prompt: str,
    length: int = 100,
    keyframe_interval: int = 5,
    keyframe_strategy: str | None = None,
    keyframe_count: int | None = None,
    keyframe_min: int | None = None,
    keyframe_max: int | None = None,
    keyframe_include_ends: bool | None = None,
    # AR generation params
    ar_temperature: float = 0.9,
    ar_top_k: int = 50,
    ar_top_p: float = 0.95,
    ar_guidance_scale: float = 2.5,
    # Diffusion params
    diff_guidance_scale: float = 2.5,
    device: str = 'cuda',
):
    """
    Complete composite generation pipeline.
    
    Args:
        prompt: Text description
        length: Target motion length in frames
        keyframe_interval: Generate keyframe every n frames (interval strategy)
        keyframe_strategy: 'interval' or 'random'
        keyframe_count: Number of random keyframes (optional)
        keyframe_min/keyframe_max: Random keyframe range (if keyframe_count is None)
        keyframe_include_ends: Always include first/last frame
        ar_*: Autoregressive generation parameters
        diff_*: Diffusion in-betweening parameters
    
    Returns:
        motion: Generated motion (T, F) in original feature space
        keyframes: The sparse keyframes from AR model (K, F)
        keyframe_indices: Indices of keyframes
    """
    print(f"Generating motion for: '{prompt}'")
    strategy = keyframe_strategy or cfg.keyframe_strategy
    k_count = keyframe_count if keyframe_count is not None else cfg.keyframe_count
    k_min = keyframe_min if keyframe_min is not None else cfg.keyframe_min
    k_max = keyframe_max if keyframe_max is not None else cfg.keyframe_max
    include_ends = keyframe_include_ends if keyframe_include_ends is not None else cfg.keyframe_include_ends

    if strategy == 'random':
        print(f"Target length: {length} frames, random keyframes")
    else:
        print(f"Target length: {length} frames, keyframe interval: {keyframe_interval}")
    
    # ========== Stage 1: AR Keyframe Generation ==========
    print("\nStage 1: Generating keyframes with MotionGPT...")
    
    # Encode text
    cond = encode_text(clip_model, [prompt])
    cond_uncond = encode_text(clip_model, [''])
    
    # Calculate how many tokens we need from VQ-VAE
    target_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate
    
    # Generate motion tokens autoregressively
    tokens = gpt.generate(
        cond=cond,
        max_new_tokens=target_tokens,
        temperature=ar_temperature,
        top_k=ar_top_k,
        top_p=ar_top_p,
        guidance_scale=ar_guidance_scale,
        cond_uncond=cond_uncond,
    )  # (1, T')
    
    # Decode tokens to continuous motion
    ar_motion_norm = vqvae.decode_indices(tokens)  # (1, T_decoded, F)
    ar_motion_norm = ar_motion_norm[0, :length]    # (T, F)
    
    print(f"AR output shape: {ar_motion_norm.shape}")
    
    # Extract sparse keyframes from AR output
    keyframe_indices = _select_keyframe_indices(
        length=length,
        keyframe_interval=keyframe_interval,
        strategy=strategy,
        keyframe_count=k_count,
        keyframe_min=k_min,
        keyframe_max=k_max,
        include_ends=include_ends,
    )
    keyframe_indices = torch.tensor(keyframe_indices, dtype=torch.long, device=device)
    
    keyframes = ar_motion_norm[keyframe_indices]  # (K, F)
    print(f"Extracted {len(keyframe_indices)} keyframes at positions: {keyframe_indices.tolist()[:10]}...")
    
    # ========== Stage 2: Diffusion In-Betweening ==========
    print("\nStage 2: Filling in-between frames with diffusion...")
    
    # Prepare for diffusion
    B, T, F = 1, length, Fdim
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    
    keyframes_batch = keyframes.unsqueeze(0)  # (1, K, F)
    keyframe_indices_batch = keyframe_indices.unsqueeze(0)  # (1, K)
    keyframe_mask_batch = torch.ones(1, len(keyframe_indices), dtype=torch.bool, device=device)
    
    # Run diffusion in-betweening
    motion_norm = diff_inbetween.sample_inbetween(
        model=inbetween_model,
        shape=(B, T, F),
        cond=cond,
        mask=mask,
        keyframes=keyframes_batch,
        keyframe_indices=keyframe_indices_batch,
        keyframe_mask=keyframe_mask_batch,
        guidance_scale=diff_guidance_scale,
        cond_uncond=cond_uncond,
    )  # (1, T, F)
    
    motion_norm = motion_norm[0]  # (T, F)
    
    # Unnormalize
    motion = motion_norm.cpu() * (std + 1e-8) + mean
    keyframes_unnorm = keyframes.cpu() * (std + 1e-8) + mean
    
    print(f"\nGeneration complete! Output shape: {motion.shape}")
    
    return motion, keyframes_unnorm, keyframe_indices.cpu()


@torch.no_grad()
def generate_ar_only(
    vqvae,
    gpt,
    clip_model,
    mean,
    std,
    cfg,
    prompt: str,
    length: int = 100,
    guidance_scale: float = 2.5,
    device: str = 'cuda',
):
    """Generate using only the AR model (no diffusion refinement)."""
    cond = encode_text(clip_model, [prompt])
    cond_uncond = encode_text(clip_model, [''])
    
    target_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate
    
    tokens = gpt.generate(
        cond=cond,
        max_new_tokens=target_tokens,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        guidance_scale=guidance_scale,
        cond_uncond=cond_uncond,
    )
    
    motion_norm = vqvae.decode_indices(tokens)
    motion_norm = motion_norm[0, :length]
    motion = motion_norm.cpu() * (std + 1e-8) + mean
    
    return motion


def main():
    """Example usage."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = CompositeConfig()
    
    # Load models
    vqvae, gpt, inbetween_model, diff_inbetween, clip_model, mean, std, Fdim = load_models(cfg, device)
    
    print("Models loaded for inference")
    
    # Generate sample motion
    prompt = "a person walks forward and then jumps"
    
    motion, keyframes, keyframe_idx = generate_composite(
        vqvae, gpt, inbetween_model, diff_inbetween, clip_model,
        mean, std, Fdim, cfg,
        prompt,
        length=120,
        keyframe_interval=cfg.keyframe_interval,
        device=device,
    )
    
    print(f"\nFinal motion shape: {motion.shape}")
    print(f"Keyframes shape: {keyframes.shape}")
    print(f"Keyframe indices: {keyframe_idx.tolist()}")
    
    # Save
    np.save("sample_composite_output.npy", motion.numpy())
    np.save("sample_composite_keyframes.npy", keyframes.numpy())
    
    print("\nSaved outputs to sample_composite_output.npy and sample_composite_keyframes.npy")


if __name__ == '__main__':
    main()
