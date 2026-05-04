# In-Between Motion Refinement (T2M-GPT + Diffusion)

This repository now focuses on a single training target:

- diffusion in-betweening with optional learned keyframe selection

AR motion generation is handled by external T2M-GPT checkpoints. This repo uses those generated motions as conditioning input and refines them with the in-between diffusion model.

## Entry Point

Use the unified CLI:

```bash
python pipeline.py <command> [options]
```

## Current Pipeline

```text
Text prompt
  -> T2M-GPT (external pretrained AR model)
  -> coarse HumanML3D motion features (263-d)
  -> keyframe selection (heuristic or learned selector)
  -> diffusion in-betweening refinement
  -> final refined motion + visualization
```

## Quick Start

### 1) Train in-between diffusion

```bash
python pipeline.py train
```

Train all selector variants in one command (heuristic, text_alignment, saliency, information_gain, retrieval_gain):

```bash
python pipeline.py train-selectors-all --force
```

Resume training from a checkpoint:

```bash
python pipeline.py train --inbetween-resume checkpoints/composite_inbetween_best.pt
```

### 2) Generate one full sample + visualization

```bash
python pipeline.py sample \
  --prompt "a person walks forward then jumps and waves" \
  --out-name demo
```

### 3) Evaluate models/selector variants

```bash
python pipeline.py evaluate --models all
```

### 4) Visualize a saved motion file

```bash
python pipeline.py visualize samples/demo_motion.npy
```

## Commands

### train

Runs in-between training via `train.py`.

```bash
python pipeline.py train
python pipeline.py train --inbetween-resume checkpoints/composite_inbetween_step100000.pt
```

Useful flags:

- `--force`
- `--inbetween-resume`
- `--inbetween-ckpt-prefix`

### train-selectors-all

Runs sequential training for all supported selector variants by temporarily patching `InbetweenTrainConfig` in `config.py` for each run and restoring the original config afterward.

```bash
python pipeline.py train-selectors-all --force
python pipeline.py train-selectors-all --only text_alignment,saliency
```

For `information_gain` and `retrieval_gain`, `train-selectors-all` now always points `selector_oracle_ckpt_path` at the sibling CondMDI checkpoint:

```text
../diffusion-motion-inbetweening/save/condmdi_randomframes/model000750000.pt
```

The shared diffusion base used for resumed selector training is still the local heuristic checkpoint.

Useful flags:

- `--force`
- `--only`
- `--keep-config`

### sample

Runs full sample generation via `run_full_sample.py`.

```bash
python pipeline.py sample --prompt "a person walks forward"
python pipeline.py sample --prompt "a person walks forward" --inbetween-ckpt ../diffusion-motion-inbetweening/save/condmdi_randomframes/model000750000.pt
```

Useful flags:

- `--inbetween-ckpt`
- `--t2mgpt-root`
- `--arlm-vq-ckpt`, `--arlm-gpt-ckpt`
- `--out-dir`, `--out-name`
- `--device`, `--stride`, `--interval-ms`

If `--inbetween-ckpt` points to a real CondMDI checkpoint such as `diffusion-motion-inbetweening/save/condmdi_randomframes/model000750000.pt`, `individual_project` will now route in-between sampling through the external CondMDI repo instead of the local reimplementation.

Current scope:

- external CondMDI is wired for inference-time sampling and evaluation paths
- local selector training remains unchanged
- when using an external CondMDI checkpoint, no learned selector is loaded from that checkpoint, so keyframe selection falls back to the heuristic path unless you add a selector separately

### arlm-generate

Generates ARLM HumanML3D feature motions only (no training side effects).

```bash
python pipeline.py arlm-generate --t2mgpt-root D:/Projects/T2M-GPT
```

Useful flags:

- `--humanml-root`
- `--split`
- `--output-dir`
- `--arlm-vq-ckpt`, `--arlm-gpt-ckpt`

### finetune (legacy helper)

Converts legacy `*_pred.npy` samples (for example from `TRAIN_ALL_GEN`) to HumanML3D feature format and launches resumed training.

```bash
python pipeline.py finetune \
  --sample-dir TRAIN_ALL_GEN \
  --resume-ckpt checkpoints/composite_inbetween_best.pt
```

If you already generate ARLM conditioning via `arlm-generate`, this helper is usually unnecessary.

### evaluate

Runs HumanML3D evaluation via `eval_ML3D.py`.

```bash
python pipeline.py evaluate --models all
python pipeline.py evaluate --models t2mgpt
python pipeline.py evaluate --models composite
```

`--models composite` expands to all composite selector variants:

- `composite_heuristic`
- `composite_text_alignment`
- `composite_information_gain`
- `composite_retrieval_gain`

Useful flags exposed in `pipeline.py`:

- `--metrics`
- `--device`
- `--humanml-root`, `--t2mgpt-root`
- `--composite-inbetween-ckpt`
- `--arlm-vq-ckpt`, `--arlm-gpt-ckpt`
- `--results-dir`, `--results-path`
- `--load-results`, `--save-results`

### visualize

```bash
python pipeline.py visualize samples/sample_motion.npy
```

## Config-Driven Settings

Most behavior is configured in `config.py`, including:

- optimizer and schedule
- keyframe selector settings
- selector mode and loss weights
- diffusion architecture and loss weights
- validation cadence and checkpoint behavior

Prefer changing defaults in `config.py` rather than adding CLI overrides.

## Recommended Workflow

```bash
# 1) Train baseline in-between model
python pipeline.py train

# 2) (Optional) Generate ARLM conditioning set
python pipeline.py arlm-generate --t2mgpt-root D:/Projects/T2M-GPT

# 3) Continue training from the best checkpoint
python pipeline.py train --inbetween-resume checkpoints/composite_inbetween_best.pt

# 4) Evaluate all selector variants + T2M-GPT
python pipeline.py evaluate --models all

# 5) Generate qualitative sample
python pipeline.py sample --prompt "a person spins and bows" --out-name qual_spin_bow
```

## Outputs

### Checkpoints

- `checkpoints/composite_inbetween_stepN.pt`
- `checkpoints/composite_inbetween_best.pt`
- selector-mode specific checkpoints (for example `composite_inbetween_information_gain_best.pt`) if trained with mode-specific prefixes

### Training logs

- `training_logs/inbetween_YYYYMMDD_HHMMSS.csv`
- `training_logs/inbetween_YYYYMMDD_HHMMSS.png`

### Sample outputs

- `samples/<name>_ar_motion.npy`
- `samples/<name>_ar_motion_native_norm.npy`
- `samples/<name>_ar_motion_local_norm.npy`
- `samples/<name>_motion.npy`
- `samples/<name>_keyframes.npy`
- `samples/<name>_keyframe_indices.npy`
- `samples/<name>_meta.json`
- `samples/<name>.gif` (or HTML fallback)

## Main Files

| File | Purpose |
|---|---|
| `pipeline.py` | Unified CLI entry point |
| `config.py` | Config defaults for training/inference/eval |
| `train.py` | In-between diffusion training loop |
| `run_full_sample.py` | End-to-end sample generation |
| `arlm_generate.py` | ARLM dataset generation |
| `eval_ML3D.py` | HumanML3D evaluation |
| `finetune_inbetween_on_arlm_samples.py` | Legacy conversion + resume helper |
| `visualize.py` | Motion rendering utilities |
| `models/diffusion.py` | Diffusion in-betweening + selector modules |

## Environment Notes

- Use a dedicated Python environment with the required scientific stack.
- For T2M-GPT integration, `--t2mgpt-root` must point to a valid checkout with pretrained weights.
- Ensure HumanML3D stats exist at `humanml/Mean.npy` and `humanml/Std.npy`.
