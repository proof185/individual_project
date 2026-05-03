"""Unified CLI for composite motion project workflows.

This wraps the existing scripts behind one intuitive command surface.
"""

import argparse
import os
import re
import subprocess
import sys
from typing import List


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


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _set_inbetween_config_fields(config_text: str, updates: dict[str, str]) -> str:
    marker_start = "@dataclass\nclass InbetweenTrainConfig:\n"
    marker_end = "\n@dataclass\nclass CompositeConfig:"

    start_idx = config_text.find(marker_start)
    end_idx = config_text.find(marker_end)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("Could not locate InbetweenTrainConfig block in config.py")

    head = config_text[:start_idx + len(marker_start)]
    body = config_text[start_idx + len(marker_start):end_idx]
    tail = config_text[end_idx:]

    for key, value in updates.items():
        pattern = rf"^(\s*{re.escape(key)}\s*:\s*[^=]+?=\s*).*$"
        replacement = rf"\1{value}"
        body, n = re.subn(pattern, replacement, body, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not update field {key!r} in InbetweenTrainConfig")

    return head + body + tail


def _quote_py_string(value: str) -> str:
    return repr(value)


def _run_train_selectors_all(args: argparse.Namespace) -> None:
    root = _project_root()
    config_path = os.path.join(root, "config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    original_config = _read_file(config_path)

    train_plan = [
        {
            "name": "heuristic",
            "selector_enabled": False,
            "selector_mode": "text_alignment",
            "ckpt_prefix": "composite_inbetween_heuristic",
            "uses_oracle": False,
        },
        {
            "name": "text_alignment",
            "selector_enabled": True,
            "selector_mode": "text_alignment",
            "ckpt_prefix": "composite_inbetween_text_alignment",
            "uses_oracle": False,
        },
        {
            "name": "saliency",
            "selector_enabled": True,
            "selector_mode": "saliency",
            "ckpt_prefix": "composite_inbetween_saliency",
            "uses_oracle": False,
        },
        {
            "name": "information_gain",
            "selector_enabled": True,
            "selector_mode": "information_gain",
            "ckpt_prefix": "composite_inbetween_information_gain",
            "uses_oracle": True,
        },
        {
            "name": "retrieval_gain",
            "selector_enabled": True,
            "selector_mode": "retrieval_gain",
            "ckpt_prefix": "composite_inbetween_retrieval_gain",
            "uses_oracle": True,
        },
    ]

    if args.only:
        requested = {name.strip().lower() for name in args.only.split(",") if name.strip()}
        if not requested:
            raise ValueError("--only was provided but no modes were parsed")
        known = {item["name"] for item in train_plan}
        unknown = requested - known
        if unknown:
            raise ValueError(
                f"Unknown modes in --only: {sorted(unknown)}. Supported: {sorted(known)}"
            )
        train_plan = [item for item in train_plan if item["name"] in requested]

    if not train_plan:
        raise ValueError("No training runs selected.")

    oracle_override = args.oracle_ckpt.strip() if args.oracle_ckpt else None
    if oracle_override:
        oracle_override = os.path.abspath(os.path.join(root, oracle_override))

    try:
        for idx, run_cfg in enumerate(train_plan, start=1):
            print(f"\n[{idx}/{len(train_plan)}] Preparing {run_cfg['name']} training run")

            selector_oracle_ckpt_value = "None"
            if run_cfg["uses_oracle"] and oracle_override:
                selector_oracle_ckpt_value = _quote_py_string(oracle_override)

            updated_config = _set_inbetween_config_fields(
                original_config,
                {
                    "use_learned_keyframe_selector": "True" if run_cfg["selector_enabled"] else "False",
                    "selector_mode": _quote_py_string(run_cfg["selector_mode"]),
                    "selector_oracle_ckpt_path": selector_oracle_ckpt_value,
                },
            )
            _write_file(config_path, updated_config)

            script_args: List[str] = ["--inbetween-ckpt-prefix", run_cfg["ckpt_prefix"]]
            if args.force:
                script_args.insert(0, "--force")

            _run("train.py", script_args)

    finally:
        if args.keep_config:
            print("Keeping config.py changes from the final run (--keep-config set).")
        else:
            _write_file(config_path, original_config)
            print("Restored original config.py")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified command runner for training, fine-tuning, generation, and visualization"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run in-between training via train.py")
    train.add_argument("--force", action="store_true")
    train.add_argument("--inbetween-resume", type=str, default=None)
    train.add_argument("--inbetween-ckpt-prefix", type=str, default=None)

    train_all = sub.add_parser(
        "train-selectors-all",
        help="Train heuristic + all learned selector modes sequentially",
    )
    train_all.add_argument("--force", action="store_true")
    train_all.add_argument(
        "--oracle-ckpt",
        type=str,
        default=None,
        help="Optional oracle checkpoint path used for information/retrieval gain runs",
    )
    train_all.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset: heuristic,text_alignment,saliency,information_gain,retrieval_gain",
    )
    train_all.add_argument(
        "--keep-config",
        action="store_true",
        help="Do not restore original config.py after completion",
    )

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
    evaluate = sub.add_parser("evaluate", help="Run HumanML3D evaluation via eval_ML3D.py")
    evaluate.add_argument("--models", type=str, default="composite",
                          help="Models to evaluate: all, composite, t2mgpt (comma-separated)")
    evaluate.add_argument("--metrics", type=str, default="all",
                          help="Metrics to compute: all or comma-separated fid,diversity,jerk,foot_skating,multimodality,multimodal_distance,matching_score,r_precision")
    evaluate.add_argument("--device", type=str, default=None)
    evaluate.add_argument("--humanml-root", type=str, default="humanml")
    evaluate.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    evaluate.add_argument("--composite-inbetween-ckpt", type=str, default=None)
    evaluate.add_argument("--arlm-vq-ckpt", type=str, default=None)
    evaluate.add_argument("--arlm-gpt-ckpt", type=str, default=None)
    evaluate.add_argument("--results-dir", type=str, default="samples/eval_results")
    evaluate.add_argument("--results-path", type=str, default=None)
    evaluate.add_argument("--load-results", action="store_true")
    evaluate.add_argument("--save-results", action="store_true")

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

    if args.command == "train-selectors-all":
        _run_train_selectors_all(args)
        return

    if args.command == "finetune":
        script_args = _args_to_script_flags(args, {"command"})
        _run("finetune_inbetween_on_arlm_samples.py", script_args)
        return

    if args.command == "arlm-generate":
        script_args = _args_to_script_flags(args, {"command"})
        _run("arlm_generate.py", script_args)
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
        _run("eval_ML3D.py", script_args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
