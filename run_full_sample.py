"""Run one full sample with T2M-GPT AR + diffusion in-betweening.

Pipeline:
1) Text -> T2M-GPT (ARLM) token generation and decode
2) Learned keyframe selector over AR motion (if present/enabled)
3) Diffusion in-betweening to produce final motion
4) Save .npy outputs and a rendered GIF/HTML visualization
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from arlm_generate_and_finetune import ARLMConfig, _load_arlm_models
from config import CompositeConfig
from generate import _select_keyframe_indices, load_models
from utils import encode_text
from visualize import animate_skeleton, recover_from_ric


def _latest_step(checkpoint_glob: str, prefix: str) -> int:
    paths = glob.glob(checkpoint_glob)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {checkpoint_glob}")

    best = -1
    for path in paths:
        base = os.path.basename(path)
        if not base.startswith(prefix):
            continue
        step_part = base[len(prefix):].removesuffix(".pt")
        if step_part.isdigit():
            best = max(best, int(step_part))

    if best < 0:
        raise RuntimeError(f"Could not parse checkpoint step from files matching: {checkpoint_glob}")
    return best


def _extract_step_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def _default_inbetween_ckpt() -> tuple[str, int | None]:
    preferred = [
        os.path.join("checkpoints", "finetuned_inbetween_best.pt"),
        os.path.join("checkpoints", "composite_inbetween_best.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path, _extract_step_from_path(path)

    step = _latest_step(
        checkpoint_glob=os.path.join("checkpoints", "composite_inbetween_step*.pt"),
        prefix="composite_inbetween_step",
    )
    return os.path.join("checkpoints", f"composite_inbetween_step{step}.pt"), step


def _resolve_arlm_ckpts(
    t2mgpt_root: str,
    vq_ckpt: str | None,
    gpt_ckpt: str | None,
) -> tuple[str, str]:
    if not vq_ckpt:
        vq_ckpt = os.path.join(t2mgpt_root, "pretrained", "VQVAE", "net_last.pth")
    if not gpt_ckpt:
        gpt_ckpt = os.path.join(t2mgpt_root, "pretrained", "VQTransformer_corruption05", "net_best_fid.pth")
    return os.path.abspath(vq_ckpt), os.path.abspath(gpt_ckpt)


def _load_arlm_stats(t2mgpt_root: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
    meta_dir = os.path.join(t2mgpt_root, "checkpoints", "t2m", "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    mean_path = os.path.join(meta_dir, "mean.npy")
    std_path = os.path.join(meta_dir, "std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            "Missing native T2M-GPT normalization stats. Expected:\n"
            f"  {mean_path}\n"
            f"  {std_path}"
        )
    mean = torch.from_numpy(np.load(mean_path)).float().to(device)
    std = torch.from_numpy(np.load(std_path)).float().to(device)
    return mean, std, meta_dir


def _generate_arlm_motion(
    prompt: str,
    clip_lib,
    clip_model,
    vq_model,
    gpt_model,
    device: torch.device,
    categorical_sample: bool,
) -> torch.Tensor:
    with torch.no_grad():
        text_tokens = clip_lib.tokenize([prompt], truncate=True).to(device)
        clip_feat = clip_model.encode_text(text_tokens).float()
        token_idx = None
        sample_modes = [bool(categorical_sample)]
        if categorical_sample:
            sample_modes.extend([True, True, False])

        for sample_mode in sample_modes:
            candidate = gpt_model.sample(clip_feat, if_categorial=sample_mode)
            if candidate is not None and candidate.ndim == 2 and candidate.shape[1] > 0:
                token_idx = candidate
                break

        if token_idx is None:
            raise RuntimeError(
                "T2M-GPT sampling produced an empty token sequence even after fallback to greedy decoding."
            )
        return vq_model.forward_decoder(token_idx)[0]


def _convert_arlm_motion_to_local_stats(
    ar_motion_native_norm: torch.Tensor,
    arlm_mean: torch.Tensor,
    arlm_std: torch.Tensor,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    clamp_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clamp_sigma > 0:
        ar_motion_native_norm = ar_motion_native_norm.clamp(-clamp_sigma, clamp_sigma)

    ar_motion_features = ar_motion_native_norm * (arlm_std + 1e-8) + arlm_mean
    ar_motion_local_norm = (ar_motion_features - local_mean) / (local_std + 1e-8)
    return ar_motion_features, ar_motion_local_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and visualize one full sample (T2M-GPT + diffusion)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate")
    parser.add_argument("--length", type=int, default=196, help="Target motion length in frames")
    parser.add_argument("--gpt-steps", type=int, default=None, help="Local composite GPT checkpoint step (only for loading pipeline config; default: latest)")
    parser.add_argument("--inbetween-steps", type=int, default=None, help="In-betweening checkpoint step (default: latest)")
    parser.add_argument("--inbetween-ckpt", type=str, default=None, help="Explicit in-betweening checkpoint path; defaults to finetuned/composite best when available")
    parser.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT", help="Path to T2M-GPT repository")
    parser.add_argument("--arlm-vq-ckpt", type=str, default=None, help="Path to T2M-GPT VQ checkpoint (default: <t2mgpt-root>/pretrained/VQVAE/net_last.pth)")
    parser.add_argument("--arlm-gpt-ckpt", type=str, default=None, help="Path to T2M-GPT GPT checkpoint (default: <t2mgpt-root>/pretrained/VQTransformer_corruption05/net_best_fid.pth)")
    parser.add_argument("--arlm-meta-dir", type=str, default=None, help="Path to T2M-GPT mean/std directory (default: <t2mgpt-root>/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta)")
    parser.add_argument("--categorical-sample", action="store_true", help="Use categorical sampling in T2M-GPT AR stage")
    parser.add_argument("--out-dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--out-name", type=str, default="full_sample", help="Output name prefix")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda or cpu")
    parser.add_argument("--diff-guidance", type=float, default=2.5, help="Diffusion CFG scale")
    parser.add_argument("--disable-selector", action="store_true", help="Disable learned keyframe selector")
    parser.add_argument("--keyframe-strategy", type=str, choices=["interval", "random"], default=None)
    parser.add_argument("--keyframe-interval", type=int, default=5)
    parser.add_argument("--keyframe-count", type=int, default=None)
    parser.add_argument("--keyframe-min", type=int, default=None)
    parser.add_argument("--keyframe-max", type=int, default=None)
    parser.add_argument("--no-keyframe-ends", action="store_true", help="Do not force first/last frame in random keyframe strategy")
    parser.add_argument("--stride", type=int, default=1, help="Visualization frame stride")
    parser.add_argument("--interval-ms", type=int, default=80, help="Animation interval in ms")
    parser.add_argument("--ar-clamp-sigma", type=float, default=0.0,
                        help="Clamp T2M-GPT VQ output to ±N sigma. Set <=0 to disable (default: disabled).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    gpt_steps = args.gpt_steps
    if gpt_steps is None:
        gpt_steps = _latest_step(
            checkpoint_glob=os.path.join("checkpoints", "composite_gpt_step*.pt"),
            prefix="composite_gpt_step",
        )

    inbetween_ckpt_path = args.inbetween_ckpt
    inbetween_steps = args.inbetween_steps
    if inbetween_ckpt_path is None:
        inbetween_ckpt_path, inferred_step = _default_inbetween_ckpt()
        if inbetween_steps is None:
            inbetween_steps = inferred_step
    else:
        inbetween_ckpt_path = os.path.abspath(inbetween_ckpt_path)
        if inbetween_steps is None:
            inbetween_steps = _extract_step_from_path(inbetween_ckpt_path)

    if inbetween_steps is None:
        inbetween_steps = 0

    cfg = CompositeConfig(
        gpt_steps=gpt_steps,
        inbetween_steps=inbetween_steps,
    )

    device_obj = torch.device(device)

    t2mgpt_root = os.path.abspath(args.t2mgpt_root)
    arlm_vq_ckpt, arlm_gpt_ckpt = _resolve_arlm_ckpts(
        t2mgpt_root=t2mgpt_root,
        vq_ckpt=args.arlm_vq_ckpt,
        gpt_ckpt=args.arlm_gpt_ckpt,
    )

    print(f"Using device: {device}")
    print(f"Using AR model: T2M-GPT ({arlm_gpt_ckpt})")
    print(f"Using diffusion checkpoint: {inbetween_ckpt_path}")

    _, _, inbetween_model, diff_inbetween, clip_model, mean, std, fdim = load_models(
        cfg,
        device,
        inbetween_ckpt_path=inbetween_ckpt_path,
    )
    mean = mean.to(device_obj)
    std = std.to(device_obj)

    arlm_cfg = ARLMConfig()
    clip_lib_arlm, clip_model_arlm, vq_model_arlm, gpt_model_arlm = _load_arlm_models(
        t2mgpt_root=t2mgpt_root,
        cfg=arlm_cfg,
        vq_ckpt=arlm_vq_ckpt,
        gpt_ckpt=arlm_gpt_ckpt,
        device=device_obj,
    )
    if args.arlm_meta_dir:
        arlm_meta_dir = os.path.abspath(args.arlm_meta_dir)
        arlm_mean = torch.from_numpy(np.load(os.path.join(arlm_meta_dir, "mean.npy"))).float().to(device_obj)
        arlm_std = torch.from_numpy(np.load(os.path.join(arlm_meta_dir, "std.npy"))).float().to(device_obj)
    else:
        arlm_mean, arlm_std, arlm_meta_dir = _load_arlm_stats(t2mgpt_root, device_obj)

    # Stage 1: T2M-GPT AR motion in normalized HumanML3D feature space.
    ar_motion_native_norm = _generate_arlm_motion(
        prompt=args.prompt,
        clip_lib=clip_lib_arlm,
        clip_model=clip_model_arlm,
        vq_model=vq_model_arlm,
        gpt_model=gpt_model_arlm,
        device=device_obj,
        categorical_sample=bool(args.categorical_sample),
    )
    ar_motion, ar_motion_norm = _convert_arlm_motion_to_local_stats(
        ar_motion_native_norm=ar_motion_native_norm,
        arlm_mean=arlm_mean,
        arlm_std=arlm_std,
        local_mean=mean,
        local_std=std,
        clamp_sigma=float(args.ar_clamp_sigma),
    )
    native_ar_length = int(ar_motion_norm.shape[0])
    if native_ar_length != int(args.length):
        print(
            f"Requested length {int(args.length)} differs from native T2M-GPT length {native_ar_length}; "
            "using native length to avoid corrupting HumanML3D features with linear resampling."
        )

    cond = encode_text(clip_model, [args.prompt])
    cond_uncond = encode_text(clip_model, [""])

    effective_length = int(ar_motion_norm.shape[0])
    use_selector = cfg.use_learned_keyframe_selector and (not args.disable_selector)
    selector_model = getattr(inbetween_model, "keyframe_selector", None)
    selector_used = bool(use_selector and selector_model is not None)
    selector_probs = None
    selector_fallback_used = False

    strategy = args.keyframe_strategy or cfg.keyframe_strategy
    keyframe_count = args.keyframe_count if args.keyframe_count is not None else cfg.keyframe_count
    keyframe_min = args.keyframe_min if args.keyframe_min is not None else cfg.keyframe_min
    keyframe_max = args.keyframe_max if args.keyframe_max is not None else cfg.keyframe_max
    include_ends = (not args.no_keyframe_ends) if args.no_keyframe_ends else cfg.keyframe_include_ends

    if selector_used:
        selector_model.eval()
        selector_valid = torch.ones(1, effective_length, dtype=torch.bool, device=device)
        selector_probs, selector_mask_st = selector_model(
            ar_motion_norm.unsqueeze(0),
            selector_valid,
            cond=cond,
        )
        keyframe_indices = torch.nonzero(selector_mask_st[0] > 0.5, as_tuple=False).squeeze(1)
        if keyframe_indices.numel() <= 2:
            selector_fallback_used = True
            idx_list = _select_keyframe_indices(
                length=effective_length,
                keyframe_interval=args.keyframe_interval,
                strategy=strategy,
                keyframe_count=keyframe_count,
                keyframe_min=keyframe_min,
                keyframe_max=keyframe_max,
                include_ends=include_ends,
            )
            keyframe_indices = torch.tensor(idx_list, dtype=torch.long, device=device)
            probs_cpu = selector_probs[0].detach().cpu()
            print(
                f"Learned selector produced only {int((selector_probs[0] > selector_model.threshold).sum().item())} hard keyframes "
                f"(mean_prob={float(probs_cpu.mean()):.4f}, max_prob={float(probs_cpu.max()):.4f}); "
                f"falling back to {len(idx_list)} heuristic keyframes."
            )
        else:
            print(f"Selected {int(keyframe_indices.numel())} keyframes with learned selector")
    else:
        idx_list = _select_keyframe_indices(
            length=effective_length,
            keyframe_interval=args.keyframe_interval,
            strategy=strategy,
            keyframe_count=keyframe_count,
            keyframe_min=keyframe_min,
            keyframe_max=keyframe_max,
            include_ends=include_ends,
        )
        keyframe_indices = torch.tensor(idx_list, dtype=torch.long, device=device)

    print(f"Keyframe indices: {keyframe_indices.detach().cpu().tolist()}")
    keyframes_norm = ar_motion_norm[keyframe_indices]

    # Stage 2: diffusion in-betweening conditioned by AR keyframes.
    B, T, Fdim_local = 1, effective_length, fdim
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    keyframes_batch = keyframes_norm.unsqueeze(0)
    keyframe_indices_batch = keyframe_indices.unsqueeze(0)
    keyframe_mask_batch = torch.ones(1, keyframe_indices.numel(), dtype=torch.bool, device=device)

    motion_norm = diff_inbetween.sample_inbetween(
        model=inbetween_model,
        shape=(B, T, Fdim_local),
        cond=cond,
        mask=mask,
        keyframes=keyframes_batch,
        keyframe_indices=keyframe_indices_batch,
        keyframe_mask=keyframe_mask_batch,
        guidance_scale=args.diff_guidance,
        cond_uncond=cond_uncond,
    )[0]

    keyframe_idx = keyframe_indices.cpu()
    ar_motion = ar_motion.cpu()
    ar_motion_native_norm = ar_motion_native_norm.cpu()
    ar_motion_local_norm = ar_motion_norm.cpu()
    mean_cpu = mean.cpu()
    std_cpu = std.cpu()
    motion = motion_norm.cpu() * (std_cpu + 1e-8) + mean_cpu
    keyframes = keyframes_norm.cpu() * (std_cpu + 1e-8) + mean_cpu

    os.makedirs(args.out_dir, exist_ok=True)
    ar_motion_path = os.path.join(args.out_dir, f"{args.out_name}_ar_motion.npy")
    ar_motion_native_norm_path = os.path.join(args.out_dir, f"{args.out_name}_ar_motion_native_norm.npy")
    ar_motion_local_norm_path = os.path.join(args.out_dir, f"{args.out_name}_ar_motion_local_norm.npy")
    motion_path = os.path.join(args.out_dir, f"{args.out_name}_motion.npy")
    keyframes_path = os.path.join(args.out_dir, f"{args.out_name}_keyframes.npy")
    keyidx_path = os.path.join(args.out_dir, f"{args.out_name}_keyframe_indices.npy")
    meta_path = os.path.join(args.out_dir, f"{args.out_name}_meta.json")
    gif_path = os.path.join(args.out_dir, f"{args.out_name}.gif")
    html_path = os.path.join(args.out_dir, f"{args.out_name}.html")

    np.save(ar_motion_path, ar_motion.numpy())
    np.save(ar_motion_native_norm_path, ar_motion_native_norm.numpy())
    np.save(ar_motion_local_norm_path, ar_motion_local_norm.numpy())
    np.save(motion_path, motion.numpy())
    np.save(keyframes_path, keyframes.numpy())
    np.save(keyidx_path, keyframe_idx.numpy())

    metadata = {
        "prompt": args.prompt,
        "length_requested": int(args.length),
        "length_native_ar": native_ar_length,
        "length_generated": int(motion.shape[0]),
        "feature_dim": int(motion.shape[1]),
        "ar_model": "t2m-gpt",
        "ar_motion_path": os.path.abspath(ar_motion_path),
        "ar_motion_native_norm_path": os.path.abspath(ar_motion_native_norm_path),
        "ar_motion_local_norm_path": os.path.abspath(ar_motion_local_norm_path),
        "local_gpt_steps": int(gpt_steps),
        "inbetween_steps": int(inbetween_steps),
        "inbetween_ckpt": os.path.abspath(inbetween_ckpt_path),
        "t2mgpt_root": t2mgpt_root,
        "t2mgpt_vq_ckpt": os.path.abspath(arlm_vq_ckpt),
        "t2mgpt_gpt_ckpt": os.path.abspath(arlm_gpt_ckpt),
        "t2mgpt_meta_dir": arlm_meta_dir,
        "device": str(device),
        "ar_clamp_sigma": float(args.ar_clamp_sigma),
        "used_selector": selector_used,
        "selector_fallback_used": selector_fallback_used,
        "keyframe_count": int(keyframe_idx.numel()),
        "keyframe_indices": keyframe_idx.tolist(),
    }
    if selector_probs is not None:
        probs_cpu = selector_probs[0].detach().cpu().numpy()
        metadata["selector_prob_mean"] = float(probs_cpu.mean())
        metadata["selector_prob_max"] = float(probs_cpu.max())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    joints = recover_from_ric(motion.numpy(), joints_num=22)
    anim = animate_skeleton(
        joints,
        title=f"Composite: {args.prompt}",
        stride=args.stride,
        interval=args.interval_ms,
        keyframe_indices=keyframe_idx.tolist(),
    )

    saved_gif = False
    try:
        fps = max(1, int(round(1000.0 / float(args.interval_ms))))
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        saved_gif = True
    except Exception as exc:
        print(f"GIF export failed ({exc}); writing HTML animation fallback.")

    if saved_gif:
        print(f"Saved animation GIF: {gif_path}")
    else:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(anim.to_jshtml())
        print(f"Saved animation HTML: {html_path}")

    if hasattr(anim, "_fig"):
        plt.close(anim._fig)

    print("Saved outputs:")
    print(f"  ar_motion: {ar_motion_path}")
    print(f"  ar_motion_native_norm: {ar_motion_native_norm_path}")
    print(f"  ar_motion_local_norm: {ar_motion_local_norm_path}")
    print(f"  motion: {motion_path}")
    print(f"  keyframes: {keyframes_path}")
    print(f"  keyframe_indices: {keyidx_path}")
    print(f"  metadata: {meta_path}")


if __name__ == "__main__":
    main()
