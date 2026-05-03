"""Unified CLI for composite motion project workflows.

This wraps the existing scripts behind one intuitive command surface.
"""

import argparse
import os
import subprocess
import sys
from typing import List


SELECTOR_TRAIN_MODES = [
    "text_alignment",
    "saliency",
    "information_gain",
    "retrieval_gain",
]


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _run(script_name: str, script_args: List[str]) -> None:
    root = _project_root()
    script_path = os.path.join(root, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, script_path, *script_args]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def _prefer_best_checkpoint(prefix: str, step: int) -> str:
    root = _project_root()
    best_ckpt = os.path.join(root, "checkpoints", f"{prefix}_best.pt")
    step_ckpt = os.path.join(root, "checkpoints", f"{prefix}_step{int(step)}.pt")
    return best_ckpt if os.path.exists(best_ckpt) else step_ckpt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified command runner for training, fine-tuning, generation, and visualization"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run training via train.py")
    train.add_argument("--stage", choices=["all", "vqvae", "gpt", "inbetween"], default="all")
    train.add_argument("--force", action="store_true")
    train.add_argument("--vqvae-steps", type=int, default=None)
    train.add_argument("--gpt-steps", type=int, default=None)
    train.add_argument("--inbetween-steps", type=int, default=None)
    train.add_argument("--inbetween-resume", type=str, default=None)
    train.add_argument("--inbetween-ckpt-prefix", type=str, default="composite_inbetween")
    train.add_argument("--keyframe-source-dir", type=str, default=None)
    train.add_argument("--disable-selector", action="store_true")
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--scheduler-type", choices=["cosine", "constant"], default=None)
    train.add_argument("--warmup-ratio", type=float, default=None)
    train.add_argument("--min-lr-ratio", type=float, default=None)
    train.add_argument("--inbetween-lr", type=float, default=None)
    train.add_argument("--selector-lr", type=float, default=None)
    train.add_argument("--ema-decay", type=float, default=None)
    train.add_argument("--selector-curriculum-fraction", type=float, default=None)
    train.add_argument("--val-interval", type=int, default=None)
    train.add_argument("--val-batches", type=int, default=None)
    train.add_argument("--grad-accum-steps", type=int, default=None)

    sample = sub.add_parser("sample", help="Generate and visualize one sample via run_full_sample.py")
    sample.add_argument("--prompt", type=str, required=True)
    sample.add_argument("--length", type=int, default=196)
    sample.add_argument("--gpt-steps", type=int, default=None)
    sample.add_argument("--inbetween-steps", type=int, default=None)
    sample.add_argument("--inbetween-ckpt", type=str, default=None)
    sample.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    sample.add_argument("--arlm-vq-ckpt", type=str, default=None)
    sample.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    sample.add_argument("--categorical-sample", action="store_true")
    sample.add_argument("--out-dir", type=str, default="samples")
    sample.add_argument("--out-name", type=str, default="full_sample")
    sample.add_argument("--device", type=str, default=None)
    sample.add_argument("--diff-guidance", type=float, default=2.5)
    sample.add_argument("--disable-selector", action="store_true")
    sample.add_argument("--keyframe-strategy", type=str, choices=["interval", "random"], default=None)
    sample.add_argument("--keyframe-interval", type=int, default=5)
    sample.add_argument("--keyframe-count", type=int, default=None)
    sample.add_argument("--keyframe-min", type=int, default=None)
    sample.add_argument("--keyframe-max", type=int, default=None)
    sample.add_argument("--no-keyframe-ends", action="store_true")
    sample.add_argument("--stride", type=int, default=1)
    sample.add_argument("--interval-ms", type=int, default=80)
    sample.add_argument("--ar-clamp-sigma", type=float, default=3.5)

    finetune = sub.add_parser(
        "finetune",
        help="Convert ARLM *_pred.npy data and fine-tune in-between model",
    )
    finetune.add_argument("--project-root", type=str, default=".")
    finetune.add_argument("--humanml-root", type=str, default="humanml")
    finetune.add_argument("--sample-dir", type=str, default="TRAIN_ALL_GEN")
    finetune.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    finetune.add_argument("--prepared-dir", type=str, default="humanml/arlm_train_pred_joint_vecs")
    finetune.add_argument("--resume-ckpt", type=str, default="checkpoints/composite_inbetween_step100000.pt")
    finetune.add_argument("--start-step", type=int, default=100000)
    finetune.add_argument("--finetune-steps", type=int, default=20000)
    finetune.add_argument("--lr", type=float, default=1e-5)
    finetune.add_argument("--ckpt-prefix", type=str, default="fine_tuned_inbetweeening")
    finetune.add_argument("--overwrite-converted", action="store_true")
    finetune.add_argument("--disable-selector", action="store_true")
    finetune.add_argument("--prepared-only", action="store_true")

    arlm_finetune = sub.add_parser(
        "arlm-finetune",
        help="Generate ARLM features and then fine-tune in-between model",
    )
    arlm_finetune.add_argument("--project-root", type=str, default=".")
    arlm_finetune.add_argument("--humanml-root", type=str, default="humanml")
    arlm_finetune.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    arlm_finetune.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    arlm_finetune.add_argument("--output-dir", type=str, default="humanml/arlm_new_joint_vecs")
    arlm_finetune.add_argument("--arlm-vq-ckpt", type=str, default=None)
    arlm_finetune.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    arlm_finetune.add_argument("--categorical-sample", action="store_true")
    arlm_finetune.add_argument("--max-len", type=int, default=196)
    arlm_finetune.add_argument("--resume-ckpt", type=str, default="checkpoints/composite_inbetween_step100000.pt")
    arlm_finetune.add_argument("--finetune-total-steps", type=int, default=120000)
    arlm_finetune.add_argument("--finetune-ckpt-prefix", type=str, default="finetuned_inbetween")
    arlm_finetune.add_argument("--finetune-lr", type=float, default=1e-5)
    arlm_finetune.add_argument("--disable-selector", action="store_true")

    arlm_generate = sub.add_parser(
        "arlm-generate",
        help="Generate ARLM HumanML3D feature motions only",
    )
    arlm_generate.add_argument("--project-root", type=str, default=".")
    arlm_generate.add_argument("--humanml-root", type=str, default="humanml")
    arlm_generate.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    arlm_generate.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    arlm_generate.add_argument("--output-dir", type=str, default="humanml/arlm_new_joint_vecs")
    arlm_generate.add_argument("--arlm-vq-ckpt", type=str, default=None)
    arlm_generate.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    arlm_generate.add_argument("--categorical-sample", action="store_true")
    arlm_generate.add_argument("--max-len", type=int, default=196)

    visualize = sub.add_parser("visualize", help="Visualize an existing motion .npy file")
    visualize.add_argument("motion_file", type=str)
    visualize.add_argument("--normalized", action="store_true")
    visualize.add_argument("--mean-path", type=str, default=None)
    visualize.add_argument("--std-path", type=str, default=None)
    visualize.add_argument("--no-center", action="store_true")
    visualize.add_argument("--bounds-percentile", type=float, default=None)
    visualize.add_argument("--interval-ms", type=int, default=None)
    visualize.add_argument("--step-through", action="store_true")
    visualize.add_argument("--save-mp4", type=str, default=None)

    # ---- Convenience single-stage train shortcuts ----
    train_vqvae = sub.add_parser("train-vqvae", help="Train only the VQ-VAE stage")
    train_vqvae.add_argument("--vqvae-steps", type=int, default=None)
    train_vqvae.add_argument("--lr", type=float, default=None)
    train_vqvae.add_argument("--batch-size", type=int, default=None)
    train_vqvae.add_argument("--num-workers", type=int, default=None)
    train_vqvae.add_argument("--grad-accum-steps", type=int, default=None)
    train_vqvae.add_argument("--scheduler-type", choices=["cosine", "constant"], default=None)
    train_vqvae.add_argument("--force", action="store_true")

    train_gpt = sub.add_parser("train-gpt", help="Train only the GPT stage (requires a trained VQ-VAE checkpoint)")
    train_gpt.add_argument("--gpt-steps", type=int, default=None)
    train_gpt.add_argument("--lr", type=float, default=None)
    train_gpt.add_argument("--batch-size", type=int, default=None)
    train_gpt.add_argument("--num-workers", type=int, default=None)
    train_gpt.add_argument("--grad-accum-steps", type=int, default=None)
    train_gpt.add_argument("--scheduler-type", choices=["cosine", "constant"], default=None)
    train_gpt.add_argument("--force", action="store_true")

    train_diffusion = sub.add_parser("train-diffusion", help="Train only the diffusion in-betweening stage")
    train_diffusion.add_argument("--inbetween-steps", type=int, default=None)
    train_diffusion.add_argument("--inbetween-resume", type=str, default=None)
    train_diffusion.add_argument("--inbetween-ckpt-prefix", type=str, default="composite_inbetween")
    train_diffusion.add_argument("--keyframe-source-dir", type=str, default=None)
    train_diffusion.add_argument("--disable-selector", action="store_true")
    train_diffusion.add_argument("--lr", type=float, default=None)
    train_diffusion.add_argument("--inbetween-lr", type=float, default=None)
    train_diffusion.add_argument("--selector-lr", type=float, default=None)
    train_diffusion.add_argument("--batch-size", type=int, default=None)
    train_diffusion.add_argument("--num-workers", type=int, default=None)
    train_diffusion.add_argument("--grad-accum-steps", type=int, default=None)
    train_diffusion.add_argument("--scheduler-type", choices=["cosine", "constant"], default=None)
    train_diffusion.add_argument("--ema-decay", type=float, default=None)
    train_diffusion.add_argument("--val-interval", type=int, default=None)
    train_diffusion.add_argument("--val-batches", type=int, default=None)
    train_diffusion.add_argument("--force", action="store_true")

    # ---- Evaluate ----
    evaluate = sub.add_parser("evaluate", help="Run HumanML3D evaluation via eval_ML3D.py")
    evaluate.add_argument("--models", type=str, default="composite",
                          help="Models to evaluate: all, composite, gpt (comma-separated)")
    evaluate.add_argument("--metrics", type=str, default="all",
                          help="Metrics to compute: all or comma-separated fid,diversity,jerk,foot_skating,multimodality,multimodal_distance,matching_score,r_precision")
    evaluate.add_argument("--num-samples", type=int, default=256)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=1234)
    evaluate.add_argument("--device", type=str, default=None)
    evaluate.add_argument("--humanml-root", type=str, default="humanml")
    evaluate.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    evaluate.add_argument("--r-precision-top-k", type=str, default="1,2,3,5")
    evaluate.add_argument("--multimodal-repeats", type=int, default=10)
    evaluate.add_argument("--multimodal-sample-count", type=int, default=10)
    evaluate.add_argument("--results-dir", type=str, default="samples/eval_results")
    evaluate.add_argument("--results-path", type=str, default=None)
    evaluate.add_argument("--composite-gpt-steps", type=int, default=None)
    evaluate.add_argument("--composite-inbetween-steps", type=int, default=None)
    evaluate.add_argument("--composite-inbetween-ckpt", type=str, default=None)
    evaluate.add_argument("--arlm-vq-ckpt", type=str, default=None)
    evaluate.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    evaluate.add_argument("--categorical-sample", action="store_true")
    evaluate.add_argument("--ar-clamp-sigma", type=float, default=0.0)
    evaluate.add_argument("--disable-selector", action="store_true")
    evaluate.add_argument("--keyframe-strategy", type=str, choices=["interval", "random"], default=None)
    evaluate.add_argument("--keyframe-interval", type=int, default=5)
    evaluate.add_argument("--keyframe-count", type=int, default=None)
    evaluate.add_argument("--keyframe-min", type=int, default=None)
    evaluate.add_argument("--keyframe-max", type=int, default=None)
    evaluate.add_argument("--no-keyframe-ends", action="store_true")
    evaluate.add_argument("--diff-guidance", type=float, default=2.5)
    evaluate.add_argument("--load-results", action="store_true")
    evaluate.add_argument("--save-results", action="store_true")

    retrain_finetune = sub.add_parser(
        "retrain-finetune",
        help="Retrain in-betweening then fine-tune in one command",
    )
    retrain_finetune.add_argument("--project-root", type=str, default=".")
    retrain_finetune.add_argument("--humanml-root", type=str, default="humanml")
    retrain_finetune.add_argument("--sample-dir", type=str, default="TRAIN_ALL_GEN")
    retrain_finetune.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    retrain_finetune.add_argument("--prepared-dir", type=str, default="humanml/arlm_train_pred_joint_vecs")
    retrain_finetune.add_argument("--base-steps", type=int, default=200000)
    retrain_finetune.add_argument("--finetune-steps", type=int, default=40000)
    retrain_finetune.add_argument("--base-lr", type=float, default=1e-4)
    retrain_finetune.add_argument("--finetune-lr", type=float, default=1e-5)
    retrain_finetune.add_argument("--batch-size", type=int, default=32)
    retrain_finetune.add_argument("--num-workers", type=int, default=0)
    retrain_finetune.add_argument("--grad-accum-steps", type=int, default=4)
    retrain_finetune.add_argument("--base-ckpt-prefix", type=str, default="composite_inbetween")
    retrain_finetune.add_argument("--finetune-ckpt-prefix", type=str, default="finetuned_inbetween")
    retrain_finetune.add_argument("--overwrite-converted", action="store_true")
    retrain_finetune.add_argument("--disable-selector", action="store_true")

    selector_stack = sub.add_parser(
        "train-selectors-all",
        help="Train shared diffusion base, then all selector variants, then ARLM adaptation",
    )
    selector_stack.add_argument("--project-root", type=str, default=".")
    selector_stack.add_argument("--humanml-root", type=str, default="humanml")
    selector_stack.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    selector_stack.add_argument(
        "--prepared-dir",
        type=str,
        default="humanml/arlm_train_pred_joint_vecs",
        help="Prepared HumanML3D ARLM conditioning directory used for final adaptation",
    )
    selector_stack.add_argument(
        "--selector-eval-root",
        type=str,
        default="D:/Projects/T2M-GPT",
        help="T2M-GPT root used to build retrieval-gain oracle targets",
    )
    selector_stack.add_argument("--base-steps", type=int, default=5000)
    selector_stack.add_argument("--selector-steps", type=int, default=5000)
    selector_stack.add_argument("--arlm-finetune-steps", type=int, default=5000)
    selector_stack.add_argument("--base-lr", type=float, default=1e-4)
    selector_stack.add_argument("--selector-lr", type=float, default=1e-4)
    selector_stack.add_argument("--arlm-lr", type=float, default=1e-5)
    selector_stack.add_argument("--batch-size", type=int, default=32)
    selector_stack.add_argument("--num-workers", type=int, default=0)
    selector_stack.add_argument("--grad-accum-steps", type=int, default=4)
    selector_stack.add_argument("--scheduler-type", choices=["cosine", "constant"], default=None)
    selector_stack.add_argument("--warmup-ratio", type=float, default=None)
    selector_stack.add_argument("--min-lr-ratio", type=float, default=None)
    selector_stack.add_argument("--selector-oracle-timesteps", type=int, default=3)
    selector_stack.add_argument("--selector-retrieval-negatives", type=int, default=31)
    selector_stack.add_argument("--val-interval", type=int, default=None)
    selector_stack.add_argument("--val-batches", type=int, default=None)
    selector_stack.add_argument(
        "--base-ckpt-prefix",
        type=str,
        default="composite_inbetween_base",
        help="Checkpoint prefix for the shared no-selector base model",
    )
    selector_stack.add_argument(
        "--skip-arlm-finetune",
        action="store_true",
        help="Stop after GT-conditioned selector training and skip ARLM adaptation",
    )
    selector_stack.add_argument(
        "--selector-modes",
        type=str,
        default=",".join(SELECTOR_TRAIN_MODES),
        help="Comma-separated selector modes to train",
    )

    return parser


def _args_to_script_flags(args: argparse.Namespace, exclude: set[str]) -> List[str]:
    out: List[str] = []
    for key, value in vars(args).items():
        if key in exclude:
            continue

        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        out.extend([flag, str(value)])
    return out


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        script_args = _args_to_script_flags(args, {"command"})
        _run("train.py", script_args)
        return

    if args.command == "sample":
        script_args = _args_to_script_flags(args, {"command"})
        _run("run_full_sample.py", script_args)
        return

    if args.command == "finetune":
        script_args = _args_to_script_flags(args, {"command"})
        _run("finetune_inbetween_on_arlm_samples.py", script_args)
        return

    if args.command == "arlm-finetune":
        script_args = _args_to_script_flags(args, {"command"})
        _run("arlm_generate_and_finetune.py", script_args)
        return

    if args.command == "arlm-generate":
        script_args = _args_to_script_flags(args, {"command"}) + ["--generate-only"]
        _run("arlm_generate_and_finetune.py", script_args)
        return

    if args.command == "visualize":
        script_args = [args.motion_file]
        if args.normalized:
            script_args.append("--normalized")
        if args.mean_path is not None:
            script_args.extend(["--mean-path", args.mean_path])
        if args.std_path is not None:
            script_args.extend(["--std-path", args.std_path])
        if args.no_center:
            script_args.append("--no-center")
        if args.bounds_percentile is not None:
            script_args.extend(["--bounds-percentile", str(args.bounds_percentile)])
        if args.interval_ms is not None:
            script_args.extend(["--interval-ms", str(args.interval_ms)])
        if args.step_through:
            script_args.append("--step-through")
        if args.save_mp4 is not None:
            script_args.extend(["--save-mp4", args.save_mp4])
        _run("visualize.py", script_args)
        return

    if args.command == "train-vqvae":
        script_args = ["--stage", "vqvae"] + _args_to_script_flags(args, {"command"})
        _run("train.py", script_args)
        return

    if args.command == "train-gpt":
        script_args = ["--stage", "gpt"] + _args_to_script_flags(args, {"command"})
        _run("train.py", script_args)
        return

    if args.command == "train-diffusion":
        script_args = ["--stage", "inbetween"] + _args_to_script_flags(args, {"command"})
        _run("train.py", script_args)
        return

    if args.command == "evaluate":
        script_args = _args_to_script_flags(args, {"command"})
        _run("eval_ML3D.py", script_args)
        return

    if args.command == "retrain-finetune":
        # Phase 1: retrain in-betweening from scratch to base-steps.
        train_args = [
            "--stage",
            "inbetween",
            "--force",
            "--inbetween-steps",
            str(int(args.base_steps)),
            "--inbetween-ckpt-prefix",
            args.base_ckpt_prefix,
            "--lr",
            str(float(args.base_lr)),
            "--batch-size",
            str(int(args.batch_size)),
            "--num-workers",
            str(int(args.num_workers)),
            "--grad-accum-steps",
            str(int(args.grad_accum_steps)),
        ]
        if args.disable_selector:
            train_args.append("--disable-selector")
        _run("train.py", train_args)

        # Phase 2: fine-tune +N steps at lower LR from the phase-1 checkpoint.
        best_resume_ckpt = os.path.join("checkpoints", f"{args.base_ckpt_prefix}_best.pt")
        fallback_resume_ckpt = os.path.join("checkpoints", f"{args.base_ckpt_prefix}_step{int(args.base_steps)}.pt")
        resume_ckpt = best_resume_ckpt if os.path.exists(best_resume_ckpt) else fallback_resume_ckpt
        finetune_args = [
            "--project-root",
            args.project_root,
            "--humanml-root",
            args.humanml_root,
            "--sample-dir",
            args.sample_dir,
            "--t2mgpt-root",
            args.t2mgpt_root,
            "--prepared-dir",
            args.prepared_dir,
            "--resume-ckpt",
            resume_ckpt,
            "--start-step",
            str(int(args.base_steps)),
            "--finetune-steps",
            str(int(args.finetune_steps)),
            "--lr",
            str(float(args.finetune_lr)),
            "--ckpt-prefix",
            args.finetune_ckpt_prefix,
        ]
        if args.overwrite_converted:
            finetune_args.append("--overwrite-converted")
        if args.disable_selector:
            finetune_args.append("--disable-selector")
        _run("finetune_inbetween_on_arlm_samples.py", finetune_args)
        return

    if args.command == "train-selectors-all":
        selector_modes = [mode.strip().lower() for mode in args.selector_modes.split(",") if mode.strip()]
        invalid_modes = [mode for mode in selector_modes if mode not in SELECTOR_TRAIN_MODES]
        if invalid_modes:
            raise ValueError(
                f"Unknown selector modes {invalid_modes!r}. Expected subset of {SELECTOR_TRAIN_MODES}."
            )
        selector_modes = list(dict.fromkeys(selector_modes))

        common_train_args = [
            "--batch-size", str(int(args.batch_size)),
            "--num-workers", str(int(args.num_workers)),
            "--grad-accum-steps", str(int(args.grad_accum_steps)),
        ]
        if args.scheduler_type is not None:
            common_train_args.extend(["--scheduler-type", args.scheduler_type])
        if args.warmup_ratio is not None:
            common_train_args.extend(["--warmup-ratio", str(float(args.warmup_ratio))])
        if args.min_lr_ratio is not None:
            common_train_args.extend(["--min-lr-ratio", str(float(args.min_lr_ratio))])
        if args.val_interval is not None:
            common_train_args.extend(["--val-interval", str(int(args.val_interval))])
        if args.val_batches is not None:
            common_train_args.extend(["--val-batches", str(int(args.val_batches))])

        base_train_args = [
            "--stage", "inbetween",
            "--force",
            "--disable-selector",
            "--inbetween-steps", str(int(args.base_steps)),
            "--inbetween-ckpt-prefix", args.base_ckpt_prefix,
            "--lr", str(float(args.base_lr)),
            *common_train_args,
        ]
        _run("train.py", base_train_args)

        base_resume_ckpt = _prefer_best_checkpoint(args.base_ckpt_prefix, args.base_steps)
        selector_target_step = int(args.base_steps + args.selector_steps)

        for selector_mode in selector_modes:
            selector_prefix = f"composite_inbetween_{selector_mode}"
            selector_train_args = [
                "--stage", "inbetween",
                "--inbetween-resume", base_resume_ckpt,
                "--inbetween-steps", str(selector_target_step),
                "--inbetween-ckpt-prefix", selector_prefix,
                "--selector-mode", selector_mode,
                "--lr", str(float(args.selector_lr)),
                "--selector-oracle-timesteps", str(int(args.selector_oracle_timesteps)),
                *common_train_args,
            ]
            if selector_mode in {"information_gain", "retrieval_gain"}:
                selector_train_args.extend(["--selector-oracle-ckpt", base_resume_ckpt])
            if selector_mode == "retrieval_gain":
                selector_train_args.extend([
                    "--selector-eval-root", args.selector_eval_root,
                    "--selector-retrieval-negatives", str(int(args.selector_retrieval_negatives)),
                ])
            _run("train.py", selector_train_args)

            if args.skip_arlm_finetune:
                continue

            selector_resume_ckpt = _prefer_best_checkpoint(selector_prefix, selector_target_step)
            arlm_prefix = f"{selector_prefix}_arlm"
            arlm_finetune_args = [
                "--project-root", args.project_root,
                "--humanml-root", args.humanml_root,
                "--prepared-dir", args.prepared_dir,
                "--resume-ckpt", selector_resume_ckpt,
                "--start-step", str(selector_target_step),
                "--finetune-steps", str(int(args.arlm_finetune_steps)),
                "--lr", str(float(args.arlm_lr)),
                "--ckpt-prefix", arlm_prefix,
                "--prepared-only",
            ]
            _run("finetune_inbetween_on_arlm_samples.py", arlm_finetune_args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
