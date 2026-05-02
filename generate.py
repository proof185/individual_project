"""Inference functions for composite motion generation."""

import argparse
import json
import os
import random

import numpy as np
import torch

from config import CompositeConfig
from models.vqvae import MotionVQVAE
from models.gpt import MotionGPT
from models.diffusion import InbetweenDiffusion, InbetweenTransformer, KeyframeSelector
from utils import setup_clip_model, encode_text, encode_text_with_tokens


def load_models(
    cfg: CompositeConfig,
    device: str = 'cuda',
    gpt_ckpt_path: str | None = None,
    inbetween_ckpt_path: str | None = None,
    load_inbetween: bool = True,
):
    """Load trained models from checkpoints."""
    # Load normalization statistics
    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    Fdim = mean.shape[0]

    if gpt_ckpt_path is None:
        gpt_ckpt_path = f'checkpoints/composite_gpt_step{cfg.gpt_steps}.pt'
    if inbetween_ckpt_path is None:
        inbetween_ckpt_path = f'checkpoints/composite_inbetween_step{cfg.inbetween_steps}.pt'

    gpt_ckpt = None
    gpt_state = None
    inbetween_ckpt = None
    inbetween_state = None

    if os.path.exists(gpt_ckpt_path):
        gpt_ckpt = torch.load(gpt_ckpt_path, map_location=device)
        gpt_state = gpt_ckpt.get('gpt')

    if load_inbetween and os.path.exists(inbetween_ckpt_path):
        inbetween_ckpt = torch.load(inbetween_ckpt_path, map_location=device)
        inbetween_state = inbetween_ckpt.get('inbetween')

    def _count_prefix_layers(state_dict: dict, prefix: str, index_pos: int) -> int:
        idx = set()
        for k in state_dict.keys():
            if not k.startswith(prefix):
                continue
            parts = k.split('.')
            if len(parts) > index_pos and parts[index_pos].isdigit():
                idx.add(int(parts[index_pos]))
        return (max(idx) + 1) if idx else 0

    # Start from config defaults, then override from checkpoint metadata/state when present.
    gpt_d_model = cfg.d_model
    gpt_n_layers = cfg.n_layers
    gpt_n_heads = cfg.n_heads

    inbetween_d_model = cfg.inbetween_d_model
    inbetween_n_layers = cfg.inbetween_layers
    inbetween_n_heads = cfg.inbetween_heads

    if isinstance(inbetween_ckpt, dict):
        saved_cfg = inbetween_ckpt.get('cfg')
        if isinstance(saved_cfg, dict):
            gpt_d_model = int(saved_cfg.get('d_model', gpt_d_model))
            gpt_n_layers = int(saved_cfg.get('n_layers', gpt_n_layers))
            gpt_n_heads = int(saved_cfg.get('n_heads', gpt_n_heads))

            inbetween_d_model = int(saved_cfg.get('inbetween_d_model', saved_cfg.get('d_model', inbetween_d_model)))
            inbetween_n_layers = int(saved_cfg.get('inbetween_layers', saved_cfg.get('n_layers', inbetween_n_layers)))
            inbetween_n_heads = int(saved_cfg.get('inbetween_heads', saved_cfg.get('n_heads', inbetween_n_heads)))

    if isinstance(gpt_state, dict):
        if 'token_emb.weight' in gpt_state:
            gpt_d_model = int(gpt_state['token_emb.weight'].shape[1])
        counted = _count_prefix_layers(gpt_state, 'blocks.', 1)
        if counted > 0:
            gpt_n_layers = counted
        # Detect cross-attention GPT vs old prepend-token GPT.
        # Old style had no 'blocks.0.cross_attn.*' keys; new style does.
        # (No architecture change needed here – MotionGPT accepts both n_layers values.)

    if isinstance(inbetween_state, dict):
        if 'frame_in.weight' in inbetween_state:
            inbetween_d_model = int(inbetween_state['frame_in.weight'].shape[0])
        counted = _count_prefix_layers(inbetween_state, 'encoder.layers.', 2)
        if counted > 0:
            inbetween_n_layers = counted
    
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
        d_model=gpt_d_model,
        n_layers=gpt_n_layers,
        n_heads=gpt_n_heads,
        dropout=cfg.dropout,
        max_len=cfg.max_len // cfg.downsample_rate + 10,
        cond_dim=512,
    ).to(device)
    
    inbetween_model = None
    diff_inbetween = None
    if load_inbetween:
        inbetween_model = InbetweenTransformer(
            feature_dim=Fdim,
            cond_dim=512,
            d_model=inbetween_d_model,
            n_layers=inbetween_n_layers,
            n_heads=inbetween_n_heads,
            dropout=cfg.dropout,
            max_len=cfg.max_len + 10,
        ).to(device)
        diff_inbetween = InbetweenDiffusion(cfg.T_diffusion, device=device)
    
    # Load checkpoints
    if gpt_ckpt is not None:
        try:
            gpt.load_state_dict(gpt_ckpt['gpt'])
        except RuntimeError as exc:
            raise RuntimeError(
                'GPT checkpoint is incompatible with the new token-level text conditioning. '
                'Retrain GPT before generating samples with this code.'
            ) from exc
        vqvae.load_state_dict(gpt_ckpt['vqvae'])
        print(f"Loaded GPT and VQ-VAE from {gpt_ckpt_path}")
    
    if load_inbetween and inbetween_ckpt is not None:
        inbetween_state_for_infer = inbetween_ckpt.get('inbetween_ema', inbetween_ckpt['inbetween'])
        inbetween_model.load_state_dict(inbetween_state_for_infer)
        selector_state = inbetween_ckpt.get('selector_ema', inbetween_ckpt.get('selector'))
        if cfg.use_learned_keyframe_selector and selector_state is not None:
            selector_d_model = int(selector_state['frame_in.weight'].shape[0])
            selector_layers = _count_prefix_layers(selector_state, 'encoder.layers.', 2) or cfg.selector_layers
            selector_model = KeyframeSelector(
                feature_dim=Fdim,
                cond_dim=512,
                d_model=selector_d_model,
                n_layers=selector_layers,
                n_heads=cfg.selector_heads,
                dropout=cfg.selector_dropout,
                max_len=cfg.max_len + 10,
                threshold=cfg.selector_threshold,
            ).to(device)
            selector_model.load_state_dict(selector_state)
            selector_model.eval()
            inbetween_model.keyframe_selector = selector_model
            print('Loaded learned keyframe selector from in-betweening checkpoint (EMA if available)')
        print(f"Loaded in-betweening model from {inbetween_ckpt_path} (EMA if available)")
    
    # Set to eval mode
    vqvae.eval()
    gpt.eval()
    if inbetween_model is not None:
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
    ar_guidance_scale: float | None = None,
    # Diffusion params
    diff_guidance_scale: float = 2.5,
    use_learned_selector: bool | None = None,
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
    ar_guidance = cfg.guidance_scale if ar_guidance_scale is None else ar_guidance_scale

    if strategy == 'random':
        print(f"Target length: {length} frames, random keyframes")
    else:
        print(f"Target length: {length} frames, keyframe interval: {keyframe_interval}")
    
    # ========== Stage 1: AR Keyframe Generation ==========
    print("\nStage 1: Generating keyframes with MotionGPT...")
    
    # Encode text
    pooled_cond, cond, cond_mask = encode_text_with_tokens(clip_model, [prompt])
    pooled_uncond, cond_uncond, cond_uncond_mask = encode_text_with_tokens(clip_model, [''])
    
    # Calculate how many tokens we need from VQ-VAE
    target_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate
    
    # Generate motion tokens autoregressively
    tokens = gpt.generate(
        cond=cond,
        max_new_tokens=target_tokens,
        temperature=ar_temperature,
        top_k=ar_top_k,
        top_p=ar_top_p,
        guidance_scale=ar_guidance,
        cond_uncond=cond_uncond,
        cond_mask=cond_mask,
        cond_uncond_mask=cond_uncond_mask,
    )  # (1, T')
    
    # Decode tokens to continuous motion
    ar_motion_norm = vqvae.decode_indices(tokens)  # (1, T_decoded, F)
    ar_motion_norm = ar_motion_norm[0]             # (T_decoded, F)

    # GPT may emit EOS early, yielding fewer frames than requested.
    # Use the effective decoded length for downstream keyframe indexing.
    effective_length = min(length, ar_motion_norm.shape[0])
    if effective_length <= 0:
        raise RuntimeError('Composite AR stage produced empty motion.')
    ar_motion_norm = ar_motion_norm[:effective_length]  # (T_eff, F)
    
    print(f"AR output shape: {ar_motion_norm.shape}")
    
    use_selector = cfg.use_learned_keyframe_selector if use_learned_selector is None else use_learned_selector
    selector_model = getattr(inbetween_model, 'keyframe_selector', None)

    if use_selector and selector_model is not None:
        selector_model.eval()
        selector_valid = torch.ones(1, effective_length, dtype=torch.bool, device=device)
        _, selector_mask_st = selector_model(
            ar_motion_norm.unsqueeze(0),
            selector_valid,
            cond=pooled_cond,
        )
        keyframe_indices = torch.nonzero(selector_mask_st[0] > 0.5, as_tuple=False).squeeze(1)
        if keyframe_indices.numel() == 0:
            keyframe_indices = torch.tensor([0, effective_length - 1], dtype=torch.long, device=device)
        print(f"Selected {int(keyframe_indices.numel())} keyframes with learned selector")
    else:
        # Extract sparse keyframes from AR output
        keyframe_indices = _select_keyframe_indices(
            length=effective_length,
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
    B, T, F = 1, effective_length, Fdim
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    
    keyframes_batch = keyframes.unsqueeze(0)  # (1, K, F)
    keyframe_indices_batch = keyframe_indices.unsqueeze(0)  # (1, K)
    keyframe_mask_batch = torch.ones(1, len(keyframe_indices), dtype=torch.bool, device=device)
    
    # Run diffusion in-betweening
    motion_norm = diff_inbetween.sample_inbetween(
        model=inbetween_model,
        shape=(B, T, F),
        cond=pooled_cond,
        mask=mask,
        keyframes=keyframes_batch,
        keyframe_indices=keyframe_indices_batch,
        keyframe_mask=keyframe_mask_batch,
        guidance_scale=diff_guidance_scale,
        cond_uncond=pooled_uncond,
    )  # (1, T, F)
    
    motion_norm = motion_norm[0]  # (T, F)

    # Clamp diffusion output to prevent physically implausible joint reconstructions
    # OOD features produce impossible skeleton positions when recovered via recover_from_ric()
    
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
    guidance_scale: float | None = None,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = 'cuda',
):
    """Generate using only the AR model (no diffusion refinement)."""
    _, cond, cond_mask = encode_text_with_tokens(clip_model, [prompt])
    _, cond_uncond, cond_uncond_mask = encode_text_with_tokens(clip_model, [''])
    guidance = cfg.guidance_scale if guidance_scale is None else guidance_scale
    
    target_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate
    
    tokens = gpt.generate(
        cond=cond,
        max_new_tokens=target_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        guidance_scale=guidance,
        cond_uncond=cond_uncond,
        cond_mask=cond_mask,
        cond_uncond_mask=cond_uncond_mask,
    )
    
    motion_norm = vqvae.decode_indices(tokens)
    motion_norm = motion_norm[0, :length]
    motion = motion_norm.cpu() * (std + 1e-8) + mean
    
    return motion


def main():
    parser = argparse.ArgumentParser(description='Generate motion using AR-only or full (AR + diffusion) pipeline')
    parser.add_argument('--mode', choices=['ar', 'full'], default='full', help='Generation mode')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--length', type=int, default=196, help='Target motion length in frames')
    parser.add_argument('--device', type=str, default=None, help='Device, e.g. cuda or cpu')
    parser.add_argument('--out-dir', type=str, default='samples', help='Output directory')
    parser.add_argument('--out-name', type=str, default='sample', help='Output name prefix')

    # Checkpoint/config controls
    parser.add_argument('--gpt-steps', type=int, default=None, help='GPT step used to build default checkpoint path')
    parser.add_argument('--inbetween-steps', type=int, default=None, help='In-between step used to build default checkpoint path')
    parser.add_argument('--gpt-ckpt', type=str, default=None, help='Explicit GPT checkpoint path')
    parser.add_argument('--inbetween-ckpt', type=str, default=None, help='Explicit in-between checkpoint path (full mode only)')

    # AR decoding controls
    parser.add_argument('--ar-guidance', type=float, default=None, help='CFG scale for AR generation (defaults to config guidance_scale)')
    parser.add_argument('--ar-temperature', type=float, default=0.9, help='Sampling temperature for AR generation')
    parser.add_argument('--ar-top-k', type=int, default=50, help='Top-k sampling cutoff for AR generation')
    parser.add_argument('--ar-top-p', type=float, default=0.95, help='Top-p sampling cutoff for AR generation')

    # Full mode keyframe/diffusion controls
    parser.add_argument('--keyframe-strategy', type=str, choices=['interval', 'random'], default=None)
    parser.add_argument('--keyframe-interval', type=int, default=5)
    parser.add_argument('--keyframe-count', type=int, default=None)
    parser.add_argument('--keyframe-min', type=int, default=None)
    parser.add_argument('--keyframe-max', type=int, default=None)
    parser.add_argument('--no-keyframe-ends', action='store_true')
    parser.add_argument('--diff-guidance', type=float, default=2.5, help='CFG scale for diffusion in-betweening')
    parser.add_argument('--disable-selector', action='store_true', help='Disable learned keyframe selector in full mode')

    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = CompositeConfig(
        keyframe_strategy='random',
        keyframe_max=20,
        keyframe_min=3,
    )
    if args.gpt_steps is not None:
        cfg.gpt_steps = int(args.gpt_steps)
    if args.inbetween_steps is not None:
        cfg.inbetween_steps = int(args.inbetween_steps)

    use_full = args.mode == 'full'
    gpt_ckpt_path = args.gpt_ckpt
    inbetween_ckpt_path = args.inbetween_ckpt

    vqvae, gpt, inbetween_model, diff_inbetween, clip_model, mean, std, Fdim = load_models(
        cfg,
        device=device,
        gpt_ckpt_path=gpt_ckpt_path,
        inbetween_ckpt_path=inbetween_ckpt_path,
        load_inbetween=use_full,
    )

    print(f'Models loaded for inference (mode={args.mode}, device={device})')

    os.makedirs(args.out_dir, exist_ok=True)
    motion_path = os.path.join(args.out_dir, f'{args.out_name}_motion.npy')
    meta_path = os.path.join(args.out_dir, f'{args.out_name}_meta.json')

    if use_full:
        motion, keyframes, keyframe_idx = generate_composite(
            vqvae,
            gpt,
            inbetween_model,
            diff_inbetween,
            clip_model,
            mean,
            std,
            Fdim,
            cfg,
            prompt=args.prompt,
            length=args.length,
            keyframe_interval=args.keyframe_interval,
            keyframe_strategy=args.keyframe_strategy,
            keyframe_count=args.keyframe_count,
            keyframe_min=args.keyframe_min,
            keyframe_max=args.keyframe_max,
            keyframe_include_ends=(not args.no_keyframe_ends),
            ar_temperature=args.ar_temperature,
            ar_top_k=args.ar_top_k,
            ar_top_p=args.ar_top_p,
            ar_guidance_scale=args.ar_guidance,
            diff_guidance_scale=args.diff_guidance,
            use_learned_selector=(not args.disable_selector),
            device=device,
        )
        keyframes_path = os.path.join(args.out_dir, f'{args.out_name}_keyframes.npy')
        key_idx_path = os.path.join(args.out_dir, f'{args.out_name}_keyframe_indices.npy')
        np.save(keyframes_path, keyframes.numpy())
        np.save(key_idx_path, keyframe_idx.numpy())
        print(f'Saved full-mode keyframes: {keyframes_path}')
        print(f'Saved full-mode keyframe indices: {key_idx_path}')
    else:
        motion = generate_ar_only(
            vqvae,
            gpt,
            clip_model,
            mean,
            std,
            cfg,
            prompt=args.prompt,
            length=args.length,
            guidance_scale=args.ar_guidance,
            temperature=args.ar_temperature,
            top_k=args.ar_top_k,
            top_p=args.ar_top_p,
            device=device,
        )

    np.save(motion_path, motion.numpy())
    print(f'Saved motion: {motion_path}')
    print(f'Final motion shape: {tuple(motion.shape)}')

    meta = {
        'mode': args.mode,
        'prompt': args.prompt,
        'length': int(motion.shape[0]),
        'device': device,
        'gpt_steps': int(cfg.gpt_steps),
        'inbetween_steps': int(cfg.inbetween_steps),
        'gpt_ckpt': gpt_ckpt_path,
        'inbetween_ckpt': inbetween_ckpt_path if use_full else None,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved metadata: {meta_path}')


if __name__ == '__main__':
    main()
