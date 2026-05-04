# Individual Project: Motion In-Betweening + Selector Training

This repository contains the local training/evaluation glue around:

- external T2M-GPT for AR motion generation
- external CondMDI checkpoints for reconstruction-based selector training
- local selector/baseline utilities for sparse keyframe conditioning

## Current Status (May 2026)

The README previously described a broader end-to-end pipeline than what exists in this checkout.
This version documents the code exactly as it is now.

### What currently works in this repo

- reconstruction selector training via `train_reconstruction.py`
- reconstruction selector command via `pipeline.py train-reconstruction`
- motion visualization via `visualize.py` / `pipeline.py visualize`

### What is currently incomplete or externally required

- `pipeline.py train` has been removed from the CLI
- `run_full_sample.py` and `eval_ML3D.py` import `arlm_generate` symbols, but `arlm_generate.py` is not present in this checkout
- `pipeline.py arlm-generate` and `pipeline.py finetune` reference scripts that are not present locally

If you plan to use sample generation/evaluation from this repo directly, restore the missing scripts or vendor those imports from your external codebase first.

## Entry Point

```bash
python pipeline.py <command> [options]
```

## Command Reference

### `train-reconstruction`

Trains the single learned reconstruction selector against a fixed external CondMDI checkpoint.

```bash
python pipeline.py train-reconstruction --force
python pipeline.py train-reconstruction --resume checkpoints/composite_selector_reconstruction_best.pt
```

Useful flags:

- `--force`
- `--resume`
- `--condmdi-ckpt`
- `--device`

- The reconstruction selector requires a CondMDI checkpoint at:
  `../diffusion-motion-inbetweening/save/condmdi_randomframes/model000750000.pt`
- Override it with `--condmdi-ckpt` or `ReconstructionTrainConfig.external_inbetween_ckpt_path`.

### `visualize`

Visualizes an existing motion `.npy` file.

```bash
python pipeline.py visualize samples/example_motion.npy
```

Common options:

- `--normalized`
- `--mean-path`
- `--std-path`
- `--interval-ms`
- `--save-mp4`

### `sample` (depends on missing local module)

Routes to `run_full_sample.py`.

```bash
python pipeline.py sample --prompt "a person walks forward" --out-name demo
```

Current caveat: `run_full_sample.py` imports `ARLMConfig` / `_load_arlm_models` from `arlm_generate`, which is not present in this repository tree.

### `evaluate` (depends on missing local module)

Routes to `eval_ML3D.py`.

```bash
python pipeline.py evaluate --models composite
```

Current caveat: `eval_ML3D.py` imports `ARLMConfig` / `_load_arlm_models` from `arlm_generate`, which is not present in this repository tree.

### `train` (removed)

The `train` subcommand is no longer exposed by `pipeline.py`.

### `arlm-generate` and `finetune` (script targets missing)

These commands are still declared in `pipeline.py`, but their target scripts are not present in this checkout:

- `arlm_generate.py`
- `finetune_inbetween_on_arlm_samples.py`

## Direct Script Usage

For selector-only training, you can call the trainer directly:

```bash
python train_reconstruction.py
```

Useful optional flags:

- `--force`
- `--resume <checkpoint.pt>`
- `--condmdi-ckpt <checkpoint.pt>`
- `--device <device>`

Checkpoint names are always derived from `selector_mode` in `ReconstructionTrainConfig`
(for example `composite_selector_reconstruction_*`).

## Configuration

Core runtime defaults live in `config.py`:

- `ReconstructionTrainConfig`: reconstruction selector training setup
- `InbetweenTrainConfig`: legacy/full in-between training config block
- `CompositeConfig`: inference/eval/default model settings

For this project state, prefer changing selector-related defaults in `ReconstructionTrainConfig`.

## Outputs

Typical artifacts produced by selector training:

- checkpoints under `checkpoints/` (for example `composite_selector_<mode>_best.pt`)
- logs in `training_logs/` (CSV and convergence plot)

Visualization outputs depend on CLI args and may include rendered files such as MP4.

## Main Files

| File | Purpose |
|---|---|
| `pipeline.py` | Unified CLI and command dispatch |
| `train_reconstruction.py` | Reconstruction selector training loop (external CondMDI-backed) |
| `config.py` | Dataclass configs for training/eval/inference |
| `run_full_sample.py` | Sample generation entry (currently depends on missing local `arlm_generate`) |
| `eval_ML3D.py` | Evaluation entry (currently depends on missing local `arlm_generate`) |
| `visualize.py` | Motion visualization utilities |
| `condmdi_adapter.py` | Adapter for external CondMDI checkpoint usage |
| `models/selector_modules/` | Selector implementations and factory (`build_keyframe_selector`) |

## Environment Notes

- Use a dedicated Python environment with the required scientific stack.
- For T2M-GPT integration, `--t2mgpt-root` must point to a valid checkout with pretrained weights.
- Ensure HumanML3D stats exist at `humanml/Mean.npy` and `humanml/Std.npy`.
