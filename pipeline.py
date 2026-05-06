"""Unified CLI for composite motion project workflows.

This wraps the existing scripts behind one intuitive command surface.
"""

import argparse
import os
import subprocess
import sys
from typing import List


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _preferred_python_executable(root: str) -> str:
    venv_python = os.path.join(root, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def _run(script_name: str, script_args: List[str]) -> None:
    root = _project_root()
    script_path = os.path.join(root, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [_preferred_python_executable(root), script_path, *script_args]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified command runner for training, fine-tuning, generation, and visualization"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_reconstruction = sub.add_parser(
        "train-reconstruction",
        help="Train the reconstruction keyframe selector against external CondMDI",
    )
    train_reconstruction.add_argument("--force", action="store_true")
    train_reconstruction.add_argument("--resume", type=str, default=None)
    train_reconstruction.add_argument("--condmdi-ckpt", type=str, default=None)
    train_reconstruction.add_argument("--device", type=str, default=None)

    sample = sub.add_parser("sample", help="Generate and visualize one sample via run_full_sample.py")
    sample.add_argument("--prompt", type=str, required=True)
    sample.add_argument("--inbetween-ckpt", type=str, default=None)
    sample.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    sample.add_argument("--arlm-vq-ckpt", type=str, default=None)
    sample.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    sample.add_argument("--out-dir", type=str, default="samples")
    sample.add_argument("--out-name", type=str, default="full_sample")
    sample.add_argument("--device", type=str, default=None)
    sample.add_argument("--stride", type=int, default=1)
    sample.add_argument("--interval-ms", type=int, default=80)

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
    finetune.add_argument("--ckpt-prefix", type=str, default="finetuned_inbetween")
    finetune.add_argument("--overwrite-converted", action="store_true")
    finetune.add_argument("--prepared-only", action="store_true")

    arlm_generate = sub.add_parser(
        "arlm-generate",
        help="Generate ARLM HumanML3D feature motions only",
    )
    arlm_generate.add_argument("--humanml-root", type=str, default="humanml")
    arlm_generate.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    arlm_generate.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    arlm_generate.add_argument("--output-dir", type=str, default="humanml/arlm_new_joint_vecs")
    arlm_generate.add_argument("--arlm-vq-ckpt", type=str, default=None)
    arlm_generate.add_argument("--arlm-gpt-ckpt", type=str, default=None)

    compare_selectors = sub.add_parser(
        "compare-selectors",
        help="Run every selector strategy on the same prompt and AR motion for side-by-side comparison",
    )
    compare_selectors.add_argument("--prompt", type=str, required=True)
    compare_selectors.add_argument(
        "--selectors",
        type=str,
        default=None,
        help="Comma-separated selector modes to run (default: all)",
    )
    compare_selectors.add_argument("--budget-ratio", type=float, default=None)
    compare_selectors.add_argument("--inbetween-ckpt", type=str, default=None)
    compare_selectors.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    compare_selectors.add_argument("--arlm-vq-ckpt", type=str, default=None)
    compare_selectors.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    compare_selectors.add_argument("--out-dir", type=str, default="samples/compare_selectors")
    compare_selectors.add_argument("--out-name", type=str, default="compare")
    compare_selectors.add_argument("--device", type=str, default=None)
    compare_selectors.add_argument("--stride", type=int, default=None)
    compare_selectors.add_argument("--interval-ms", type=int, default=None)

    ghost = sub.add_parser(
        "ghost",
        help="Render a static ghost-trail image of a motion sequence for dissertation/paper figures",
    )
    ghost.add_argument("motion_file", type=str)
    ghost.add_argument("--keyframe-indices", type=str, default=None)
    ghost.add_argument("--n-ghosts", type=int, default=None)
    ghost.add_argument("--stride", type=int, default=None)
    ghost.add_argument("--spread", type=float, default=None)
    ghost.add_argument("--out", type=str, default=None)
    ghost.add_argument("--elev", type=float, default=None)
    ghost.add_argument("--azim", type=float, default=None)
    ghost.add_argument("--figsize", type=str, default=None)
    ghost.add_argument("--dpi", type=int, default=None)
    ghost.add_argument("--bone-lw", type=float, default=None)
    ghost.add_argument("--joint-size", type=float, default=None)
    ghost.add_argument("--regular-color", type=str, default=None)
    ghost.add_argument("--keyframe-color", type=str, default=None)
    ghost.add_argument("--no-floor", action="store_true")
    ghost.add_argument("--floor-tiles", type=int, default=None)
    ghost.add_argument("--show-axes", action="store_true")
    ghost.add_argument("--title", type=str, default=None)
    ghost.add_argument("--normalized", action="store_true")
    ghost.add_argument("--mean-path", type=str, default=None)
    ghost.add_argument("--std-path", type=str, default=None)

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

    # ---- Evaluate ----
    evaluate = sub.add_parser("evaluate", help="Run HumanML3D evaluation via eval.py")

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

    if args.command == "sample":
        script_args = _args_to_script_flags(args, {"command"})
        _run("run_full_sample.py", script_args)
        return

    if args.command == "train-reconstruction":
        script_args = _args_to_script_flags(args, {"command"})
        _run("train_reconstruction.py", script_args)
        return

    if args.command == "finetune":
        script_args = _args_to_script_flags(args, {"command"})
        _run("finetune_inbetween_on_arlm_samples.py", script_args)
        return

    if args.command == "arlm-generate":
        script_args = _args_to_script_flags(args, {"command"})
        _run("arlm_generate.py", script_args)
        return

    if args.command == "compare-selectors":
        script_args = _args_to_script_flags(args, {"command"})
        _run("run_compare_selectors.py", script_args)
        return

    if args.command == "ghost":
        script_args = [args.motion_file]
        script_args += _args_to_script_flags(args, {"command", "motion_file"})
        _run("visualize_ghost.py", script_args)
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

    if args.command == "evaluate":
        script_args = _args_to_script_flags(args, {"command"})
        _run("eval.py", script_args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
