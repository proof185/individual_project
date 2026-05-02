"""Generate ARLM motions and fine-tune in-between diffusion from a resume checkpoint.

This script does two stages:
1) Use T2M-GPT (ARLM) to generate one motion per HumanML3D id and save
   feature-space motions (263-dim joint vectors) to an output directory.
2) Launch this repo's in-between diffusion training using those generated
   motions as keyframe conditioning input.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
import types

import numpy as np
import torch


@dataclass
class ARLMConfig:
    nb_code: int = 512
    code_dim: int = 512
    output_emb_width: int = 512
    down_t: int = 2
    stride_t: int = 2
    width: int = 512
    depth: int = 3
    dilation_growth_rate: int = 3
    embed_dim_gpt: int = 1024
    clip_dim: int = 512
    block_size: int = 51
    num_layers: int = 9
    n_head_gpt: int = 16
    drop_out_rate: float = 0.1
    ff_rate: int = 4
    quantizer: str = "ema_reset"
    mu: float = 0.99


def _read_ids(split_file: str) -> list[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _pick_caption(text_path: str) -> str:
    if not os.path.exists(text_path):
        return ""
    best_caption = ""
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            caption = parts[0].strip()
            if not caption:
                continue
            if len(parts) >= 4:
                try:
                    f_tag = float(parts[2])
                    to_tag = float(parts[3])
                    if (np.isnan(f_tag) or f_tag == 0.0) and (np.isnan(to_tag) or to_tag == 0.0):
                        return caption
                except Exception:
                    pass
            if not best_caption:
                best_caption = caption
    return best_caption


def _align_length(motion: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return motion[:1]
    if motion.shape[0] == target_len:
        return motion
    if motion.shape[0] <= 1:
        return np.repeat(motion[:1], target_len, axis=0)
    x_old = np.linspace(0.0, 1.0, motion.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.stack([np.interp(x_new, x_old, motion[:, j]) for j in range(motion.shape[1])], axis=1)
    return out.astype(np.float32)


def _validate_humanml_features(motion: np.ndarray, expected_dim: int, sample_id: str) -> None:
    if motion.ndim != 2:
        raise ValueError(
            f"Expected generated HumanML3D features with shape (T, {expected_dim}) for {sample_id}, got {motion.shape}"
        )
    if motion.shape[1] != expected_dim:
        raise ValueError(
            f"Expected generated HumanML3D feature dim {expected_dim} for {sample_id}, got {motion.shape[1]}"
        )


def _load_arlm_models(t2mgpt_root: str, cfg: ARLMConfig, vq_ckpt: str, gpt_ckpt: str, device: torch.device):
    # Avoid module namespace collisions with this repo's local `models` package.
    for mod_name in ["models.t2m_trans", "models.vqvae", "models"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    sys.path.insert(0, t2mgpt_root)

    # T2M-GPT uses imports like `models.t2m_trans` but `models` has no __init__.py.
    # Create a namespace package so imports resolve reliably even when another `models`
    # package exists in the environment.
    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = [os.path.join(t2mgpt_root, "models")]  # type: ignore[attr-defined]
    sys.modules["models"] = models_pkg

    import clip  # type: ignore
    import models.t2m_trans as t2m_trans  # type: ignore
    import models.vqvae as t2m_vqvae  # type: ignore

    arlm_args = SimpleNamespace(
        dataname="t2m",
        quantizer=cfg.quantizer,
        mu=cfg.mu,
    )

    vq_model = t2m_vqvae.HumanVQVAE(
        arlm_args,
        cfg.nb_code,
        cfg.code_dim,
        cfg.output_emb_width,
        cfg.down_t,
        cfg.stride_t,
        cfg.width,
        cfg.depth,
        cfg.dilation_growth_rate,
    ).to(device)

    gpt_model = t2m_trans.Text2Motion_Transformer(
        num_vq=cfg.nb_code,
        embed_dim=cfg.embed_dim_gpt,
        clip_dim=cfg.clip_dim,
        block_size=cfg.block_size,
        num_layers=cfg.num_layers,
        n_head=cfg.n_head_gpt,
        drop_out_rate=cfg.drop_out_rate,
        fc_rate=cfg.ff_rate,
    ).to(device)

    vq_state = torch.load(vq_ckpt, map_location="cpu")
    gpt_state = torch.load(gpt_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_state["net"], strict=True)
    gpt_model.load_state_dict(gpt_state["trans"], strict=True)
    vq_model.eval()
    gpt_model.eval()

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip, clip_model, vq_model, gpt_model


def generate_arlm_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ARLM generation device: {device}")

    humanml_root = os.path.abspath(args.humanml_root)
    t2mgpt_root = os.path.abspath(args.t2mgpt_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    split_file = os.path.join(humanml_root, f"{args.split}.txt")
    text_dir = os.path.join(humanml_root, "texts")
    gt_motion_dir = os.path.join(humanml_root, "new_joint_vecs")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Missing split file: {split_file}")
    if not os.path.exists(text_dir):
        raise FileNotFoundError(f"Missing text dir: {text_dir}")
    if not os.path.exists(gt_motion_dir):
        raise FileNotFoundError(f"Missing gt motion dir: {gt_motion_dir}")

    ids = _read_ids(split_file)
    arlm_cfg = ARLMConfig()

    arlm_vq_ckpt = args.arlm_vq_ckpt
    if not arlm_vq_ckpt:
        arlm_vq_ckpt = os.path.join(t2mgpt_root, "pretrained", "VQVAE", "net_last.pth")
    arlm_gpt_ckpt = args.arlm_gpt_ckpt
    if not arlm_gpt_ckpt:
        arlm_gpt_ckpt = os.path.join(t2mgpt_root, "pretrained", "VQTransformer_corruption05", "net_best_fid.pth")

    arlm_vq_ckpt = os.path.abspath(arlm_vq_ckpt)
    arlm_gpt_ckpt = os.path.abspath(arlm_gpt_ckpt)

    if not os.path.exists(arlm_vq_ckpt) or not os.path.exists(arlm_gpt_ckpt):
        raise FileNotFoundError(
            "Missing ARLM pretrained checkpoints. Expected:\n"
            f"  VQ: {arlm_vq_ckpt}\n"
            f"  GPT: {arlm_gpt_ckpt}\n"
            "Download them from T2M-GPT README by running in T2M-GPT:\n"
            "  bash dataset/prepare/download_model.sh\n"
            "or pass explicit --arlm-vq-ckpt and --arlm-gpt-ckpt."
        )

    clip_lib, clip_model, vq_model, gpt_model = _load_arlm_models(
        t2mgpt_root,
        arlm_cfg,
        arlm_vq_ckpt,
        arlm_gpt_ckpt,
        device,
    )

    denorm_mean = np.load(os.path.join(humanml_root, "Mean.npy")).astype(np.float32)
    denorm_std = np.load(os.path.join(humanml_root, "Std.npy")).astype(np.float32)
    feature_dim = int(denorm_mean.shape[0])

    saved = 0
    skipped = 0
    with torch.no_grad():
        for i, mid in enumerate(ids, start=1):
            text_path = os.path.join(text_dir, f"{mid}.txt")
            gt_path = os.path.join(gt_motion_dir, f"{mid}.npy")
            if not os.path.exists(gt_path):
                skipped += 1
                continue

            caption = _pick_caption(text_path)
            if not caption:
                skipped += 1
                continue

            gt_motion = np.load(gt_path).astype(np.float32)
            target_len = min(int(gt_motion.shape[0]), int(args.max_len))
            if target_len < 2:
                skipped += 1
                continue

            text_tokens = clip_lib.tokenize([caption], truncate=True).to(device)
            clip_feat = clip_model.encode_text(text_tokens).float()

            token_idx = gpt_model.sample(clip_feat, if_categorial=bool(args.categorical_sample))
            pred_norm = vq_model.forward_decoder(token_idx)[0].detach().cpu().numpy().astype(np.float32)
            pred_denorm = pred_norm * denorm_std + denorm_mean
            _validate_humanml_features(pred_denorm, feature_dim, mid)
            pred_aligned = _align_length(pred_denorm, target_len)
            _validate_humanml_features(pred_aligned, feature_dim, mid)

            np.save(os.path.join(output_dir, f"{mid}.npy"), pred_aligned)
            saved += 1

            if i % 200 == 0:
                print(f"Generated {i}/{len(ids)} ids (saved={saved}, skipped={skipped})")

    print(f"ARLM generation finished: saved={saved}, skipped={skipped}, out={output_dir}")


def run_finetune(args):
    train_py = os.path.join(os.path.abspath(args.project_root), "train.py")
    resume_ckpt = os.path.abspath(args.resume_ckpt)
    keyframe_source_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(train_py):
        raise FileNotFoundError(f"Missing train.py at: {train_py}")
    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Missing resume checkpoint: {resume_ckpt}")
    if not os.path.exists(keyframe_source_dir):
        raise FileNotFoundError(f"Missing generated ARLM dataset dir: {keyframe_source_dir}")

    cmd = [
        sys.executable,
        train_py,
        "--stage",
        "inbetween",
        "--inbetween-resume",
        resume_ckpt,
        "--inbetween-ckpt-prefix",
        args.finetune_ckpt_prefix,
        "--inbetween-steps",
        str(int(args.finetune_total_steps)),
        "--keyframe-source-dir",
        keyframe_source_dir,
        "--lr",
        str(float(args.finetune_lr)),
    ]
    if args.disable_selector:
        cmd.append("--disable-selector")

    print("Launching fine-tune command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=os.path.abspath(args.project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ARLM dataset and fine-tune in-between diffusion")
    parser.add_argument("--project-root", type=str, default=".", help="Path to individual_project root")
    parser.add_argument("--humanml-root", type=str, default="humanml", help="Path to HumanML3D root used by individual_project")
    parser.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT", help="Path to T2M-GPT repository")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split to generate ARLM conditioning for")
    parser.add_argument("--output-dir", type=str, default="humanml/arlm_new_joint_vecs", help="Where to write generated ARLM feature motions")

    parser.add_argument(
        "--arlm-vq-ckpt",
        type=str,
        default=None,
        help="Path to ARLM VQ checkpoint (default: <t2mgpt-root>/pretrained/VQVAE/net_last.pth)",
    )
    parser.add_argument(
        "--arlm-gpt-ckpt",
        type=str,
        default=None,
        help="Path to ARLM GPT checkpoint (default: <t2mgpt-root>/pretrained/VQTransformer_corruption05/net_best_fid.pth)",
    )
    parser.add_argument("--categorical-sample", action="store_true", help="Use stochastic categorical token sampling instead of greedy")
    parser.add_argument("--max-len", type=int, default=196, help="Max generated sequence length")

    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default="checkpoints/composite_inbetween_step100000.pt",
        help="In-between checkpoint to resume from",
    )
    parser.add_argument(
        "--finetune-total-steps",
        type=int,
        default=120000,
        help="Final absolute step after fine-tuning (e.g., 120000 resumes from 100k for +20k)",
    )
    parser.add_argument(
        "--finetune-ckpt-prefix",
        type=str,
        default="finetuned_inbetween",
        help="Checkpoint filename prefix for fine-tuned in-between model",
    )
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate for fine-tune")
    parser.add_argument(
        "--disable-selector",
        action="store_true",
        help="Disable learned keyframe selector during fine-tuning",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate the ARLM HumanML3D feature dataset and skip diffusion fine-tuning",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_arlm_dataset(args)
    if args.generate_only:
        print("Generation complete; skipping fine-tuning because --generate-only was set.")
        return
    run_finetune(args)


if __name__ == "__main__":
    main()
