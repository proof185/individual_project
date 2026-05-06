"""Run the full sample pipeline with every selector strategy on the same prompt.

The AR motion is generated once, then each selector strategy is applied to
produce its own diffusion in-betweened result.  Outputs land in a single
directory so visualisations can be compared side-by-side.

Pipeline per strategy
---------------------
1. (shared) Text → T2M-GPT AR motion
2. Selector-specific keyframe selection on the AR features
3. Diffusion in-betweening conditioned on those keyframes
4. Save <out_name>_<selector>_motion.npy + GIF

Usage
-----
    python run_compare_selectors.py --prompt "a person walks forward" \\
        --t2mgpt-root D:/Projects/T2M-GPT --out-dir samples/compare
"""

import argparse
import glob as _glob
import json
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from keyframe_selectors import SELECTOR_MODE_CHOICES, build_keyframe_selector
from config import CompositeConfig
from utils import encode_text, load_inbetween_model
from visualize import animate_skeleton, recover_from_ric

# Helpers sourced from eval.py which handles missing arlm_generate gracefully.
from eval import (
    _generate_arlm_motion,
    _load_arlm_runtime,
    _load_arlm_stats,
    _resolve_arlm_ckpts,
)


# Modes that score frames using raw (unnormalized) AR features.
_FEATURE_BASED_MODES = {"energy", "pose_extrema", "interpolation_error", "contact_transition"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _default_inbetween_ckpt() -> str:
    preferred = [
        os.path.join("checkpoints", "finetuned_inbetween_best.pt"),
        os.path.join("checkpoints", "composite_inbetween_best.pt"),
        os.path.join("checkpoints", "composite_selector_reconstruction_best.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    for pattern in (
        "composite_inbetween_step*.pt",
        "composite_selector_reconstruction_step*.pt",
    ):
        paths = _glob.glob(os.path.join("checkpoints", pattern))
        if paths:
            best = -1
            for p in paths:
                m = re.search(r"step(\d+)\.pt$", os.path.basename(p))
                if m:
                    best = max(best, int(m.group(1)))
            if best >= 0:
                stem = pattern.replace("step*.pt", f"step{best}.pt")
                return os.path.join("checkpoints", stem)

    raise FileNotFoundError("No inbetween checkpoint found under checkpoints/")


def _convert_arlm_motion_to_local_stats(
    ar_motion_native_norm: torch.Tensor,
    arlm_mean: torch.Tensor,
    arlm_std: torch.Tensor,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    clamp_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clamp_sigma > 0:
        ar_motion_native_norm = ar_motion_native_norm.clamp(-clamp_sigma, clamp_sigma)
    ar_motion = ar_motion_native_norm * (arlm_std + 1e-8) + arlm_mean
    ar_motion_norm = (ar_motion - local_mean) / (local_std + 1e-8)
    return ar_motion, ar_motion_norm


# ── per-selector selection ────────────────────────────────────────────────────

def _select_keyframes(
    mode: str,
    reconstruction_selector,
    ar_motion: torch.Tensor,
    ar_motion_norm: torch.Tensor,
    cond: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratio: float,
    fdim: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Return (keyframe_indices_1d, error_string_or_None)."""
    if mode == "reconstruction":
        if reconstruction_selector is None:
            return None, "no reconstruction selector loaded in checkpoint"
        reconstruction_selector.eval()
        with torch.no_grad():
            probs, hard = reconstruction_selector(ar_motion_norm.unsqueeze(0), valid_mask, cond=cond)
        indices = torch.nonzero(hard[0] > 0.5, as_tuple=False).squeeze(1)
        if indices.numel() <= 2:
            return None, (
                f"reconstruction selector produced only {int(indices.numel())} keyframes "
                f"(mean_prob={float(probs[0].mean()):.4f})"
            )
        return indices, None

    selector = build_keyframe_selector(
        mode=mode,
        feature_dim=fdim,
        budget_ratio=budget_ratio,
    ).to(device)
    selector.eval()
    motion_input = ar_motion if mode in _FEATURE_BASED_MODES else ar_motion_norm
    with torch.no_grad():
        _, hard = selector(motion_input.unsqueeze(0), valid_mask)
    indices = torch.nonzero(hard[0] > 0.5, as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        return None, f"{mode} selector returned no keyframes"
    return indices, None


# ── diffusion ─────────────────────────────────────────────────────────────────

def _run_diffusion(
    inbetween_model,
    diff_inbetween,
    ar_motion_norm: torch.Tensor,
    keyframe_indices: torch.Tensor,
    cond: torch.Tensor,
    cond_uncond: torch.Tensor,
    cfg: CompositeConfig,
    fdim: int,
    device: torch.device,
    prompt: str,
) -> torch.Tensor:
    T = int(ar_motion_norm.shape[0])
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    keyframes_norm = ar_motion_norm[keyframe_indices]
    return diff_inbetween.sample_inbetween(
        model=inbetween_model,
        shape=(1, T, fdim),
        cond=cond,
        mask=mask,
        keyframes=keyframes_norm.unsqueeze(0),
        keyframe_indices=keyframe_indices.unsqueeze(0),
        keyframe_mask=torch.ones(1, keyframe_indices.numel(), dtype=torch.bool, device=device),
        guidance_scale=cfg.guidance_scale,
        cond_uncond=cond_uncond,
        source_motion=ar_motion_norm.unsqueeze(0),
        text_prompt=prompt,
    )[0]


# ── output ────────────────────────────────────────────────────────────────────

def _save_result(
    out_dir: str,
    out_name: str,
    selector_mode: str,
    motion_norm: torch.Tensor,
    keyframe_indices: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    metadata: dict,
    stride: int,
    interval_ms: int,
    prompt: str,
) -> dict:
    prefix = os.path.join(out_dir, f"{out_name}_{selector_mode}")
    motion_path = f"{prefix}_motion.npy"
    keyidx_path = f"{prefix}_keyframe_indices.npy"
    meta_path = f"{prefix}_meta.json"
    gif_path = f"{prefix}.gif"
    html_path = f"{prefix}.html"

    motion = motion_norm.cpu() * (std.cpu() + 1e-8) + mean.cpu()
    np.save(motion_path, motion.numpy())
    np.save(keyidx_path, keyframe_indices.cpu().numpy())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    joints = recover_from_ric(motion.numpy(), joints_num=22)
    anim = animate_skeleton(
        joints,
        title=f"{selector_mode}: {prompt}",
        stride=stride,
        interval=interval_ms,
        keyframe_indices=keyframe_indices.cpu().tolist(),
    )

    saved_gif = False
    try:
        fps = max(1, int(round(1000.0 / float(interval_ms))))
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        saved_gif = True
    except Exception as exc:
        print(f"  GIF export failed ({exc}); writing HTML fallback.")
    if not saved_gif:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(anim.to_jshtml())
    if hasattr(anim, "_fig"):
        plt.close(anim._fig)

    return {
        "motion": os.path.abspath(motion_path),
        "keyframe_indices": os.path.abspath(keyidx_path),
        "metadata": os.path.abspath(meta_path),
        "animation": os.path.abspath(gif_path if saved_gif else html_path),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all selector strategies on the same prompt and AR motion for comparison"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--selectors",
        type=str,
        default=",".join(SELECTOR_MODE_CHOICES),
        help="Comma-separated selector modes (default: all)",
    )
    parser.add_argument("--budget-ratio", type=float, default=0.2)
    parser.add_argument("--inbetween-ckpt", type=str, default=None)
    parser.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    parser.add_argument("--arlm-vq-ckpt", type=str, default=None)
    parser.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    parser.add_argument("--arlm-meta-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="samples/compare_selectors")
    parser.add_argument("--out-name", type=str, default="compare")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--interval-ms", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    unknown = [s for s in selectors if s not in SELECTOR_MODE_CHOICES]
    if unknown:
        raise ValueError(f"Unknown selector modes: {unknown}. Valid: {SELECTOR_MODE_CHOICES}")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    inbetween_ckpt = os.path.abspath(args.inbetween_ckpt or _default_inbetween_ckpt())
    cfg = CompositeConfig()
    t2mgpt_root = os.path.abspath(args.t2mgpt_root)
    arlm_vq_ckpt, arlm_gpt_ckpt = _resolve_arlm_ckpts(t2mgpt_root, args.arlm_vq_ckpt, args.arlm_gpt_ckpt)

    print(f"Device         : {device_str}")
    print(f"Selectors      : {selectors}")
    print(f"AR model       : T2M-GPT ({arlm_gpt_ckpt})")
    print(f"Diffusion ckpt : {inbetween_ckpt}")
    print()

    # ── Load models once ──────────────────────────────────────────────────────
    inbetween_model, diff_inbetween, clip_model, mean, std, fdim = load_inbetween_model(
        cfg, device_str, inbetween_ckpt_path=inbetween_ckpt
    )
    mean = mean.to(device)
    std = std.to(device)

    ARLMConfig, load_arlm_models = _load_arlm_runtime(t2mgpt_root)
    arlm_cfg = ARLMConfig()
    clip_lib_arlm, clip_model_arlm, vq_model_arlm, gpt_model_arlm = load_arlm_models(
        t2mgpt_root, arlm_cfg, arlm_vq_ckpt, arlm_gpt_ckpt, device
    )

    if args.arlm_meta_dir:
        meta_dir = os.path.abspath(args.arlm_meta_dir)
        arlm_mean = torch.from_numpy(np.load(os.path.join(meta_dir, "mean.npy"))).float().to(device)
        arlm_std = torch.from_numpy(np.load(os.path.join(meta_dir, "std.npy"))).float().to(device)
    else:
        arlm_mean, arlm_std, _ = _load_arlm_stats(t2mgpt_root, device)

    # ── Stage 1: generate AR motion once ─────────────────────────────────────
    print(f'Generating AR motion for: "{args.prompt}"')
    ar_motion_native_norm = _generate_arlm_motion(
        prompt=args.prompt,
        clip_lib=clip_lib_arlm,
        clip_model=clip_model_arlm,
        vq_model=vq_model_arlm,
        gpt_model=gpt_model_arlm,
        device=device,
        categorical_sample=False,
    )
    ar_motion, ar_motion_norm = _convert_arlm_motion_to_local_stats(
        ar_motion_native_norm, arlm_mean, arlm_std, mean, std
    )
    T = int(ar_motion_norm.shape[0])
    print(f"AR motion: {T} frames, feature dim: {fdim}\n")

    cond = encode_text(clip_model, [args.prompt])
    cond_uncond = encode_text(clip_model, [""])
    valid_mask = torch.ones(1, T, dtype=torch.bool, device=device)

    reconstruction_selector = getattr(inbetween_model, "keyframe_selector", None)

    os.makedirs(args.out_dir, exist_ok=True)
    ar_path = os.path.join(args.out_dir, f"{args.out_name}_ar_motion.npy")
    np.save(ar_path, ar_motion.cpu().numpy())
    print(f"AR motion saved: {ar_path}\n")

    # ── Stage 2 & 3: per-selector ─────────────────────────────────────────────
    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    for mode in selectors:
        print(f"── {mode} ──")
        keyframe_indices, err = _select_keyframes(
            mode=mode,
            reconstruction_selector=reconstruction_selector,
            ar_motion=ar_motion,
            ar_motion_norm=ar_motion_norm,
            cond=cond,
            valid_mask=valid_mask,
            budget_ratio=args.budget_ratio,
            fdim=fdim,
            device=device,
        )
        if err is not None:
            print(f"  SKIPPED: {err}\n")
            skipped[mode] = err
            continue

        idx_list = keyframe_indices.cpu().tolist()
        print(f"  {len(idx_list)} keyframes: {idx_list}")

        motion_norm = _run_diffusion(
            inbetween_model=inbetween_model,
            diff_inbetween=diff_inbetween,
            ar_motion_norm=ar_motion_norm,
            keyframe_indices=keyframe_indices,
            cond=cond,
            cond_uncond=cond_uncond,
            cfg=cfg,
            fdim=fdim,
            device=device,
            prompt=args.prompt,
        )

        paths = _save_result(
            out_dir=args.out_dir,
            out_name=args.out_name,
            selector_mode=mode,
            motion_norm=motion_norm,
            keyframe_indices=keyframe_indices,
            mean=mean,
            std=std,
            metadata={
                "prompt": args.prompt,
                "selector_mode": mode,
                "budget_ratio": args.budget_ratio,
                "ar_length": T,
                "feature_dim": fdim,
                "keyframe_count": len(idx_list),
                "keyframe_indices": idx_list,
                "inbetween_ckpt": inbetween_ckpt,
            },
            stride=args.stride,
            interval_ms=args.interval_ms,
            prompt=args.prompt,
        )
        results[mode] = paths
        print(f"  Saved motion   : {paths['motion']}")
        print(f"  Saved animation: {paths['animation']}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'Selector':<25} {'Status'}")
    print("-" * 60)
    for mode in selectors:
        if mode in skipped:
            print(f"  {mode:<23} SKIPPED  ({skipped[mode]})")
        else:
            print(f"  {mode:<23} OK")

    summary_path = os.path.join(args.out_dir, f"{args.out_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": args.prompt,
                "selectors_run": selectors,
                "results": results,
                "skipped": skipped,
                "ar_motion": os.path.abspath(ar_path),
            },
            f,
            indent=2,
        )
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
