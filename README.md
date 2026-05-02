# Composite Motion Generation

This repository implements a **three-stage text-to-motion pipeline**:

1. **VQ-VAE** — tokenises continuous motion into discrete codes.
2. **GPT (ARLM)** — autoregressively generates a full coarse-motion token sequence from a text prompt.
3. **Diffusion in-betweening + Keyframe Selector** — the learned selector identifies the most expressive frames in the GPT output; the diffusion model then refines the full sequence by in-betweening between those keyframes.

The simplest way to use the project is through the unified entry point:

```bash
python pipeline.py <command> [options]
```

## Architecture Overview

```
Text prompt
    │
    ▼
CLIP text encoder (ViT-B/32, frozen)
    │
    ▼
MotionGPT  ──── cross-attention conditioning at every layer ─── CLIP embedding
  (512-d, 12 layers, codebook 1024)
    │  autoregressive token generation
    ▼
VQ-VAE decoder  (residual blocks, GroupNorm)
    │  coarse full-length motion
    ▼
Learned Keyframe Selector
    │  sparse keyframe mask + keyframe frames
    ▼
Diffusion In-betweening (InbetweenTransformer, 512-d 12-layer)
    │  denoising conditioned on keyframes + text
    ▼
Final refined motion (T, 263)
```

## Quick Start

### 1) Train all stages

```bash
python pipeline.py train --stage all
```

### 2) Generate one full sample + visualization

```bash
python pipeline.py sample --prompt "a person walks forward then jumps and waves" --length 120 --out-name demo
```

### 3) Visualize a saved motion file

```bash
python pipeline.py visualize samples/demo_motion.npy
```

## Unified Commands

### Train

Train a single stage or all stages sequentially:

```bash
# All stages in sequence
python pipeline.py train --stage all

# Individual stages
python pipeline.py train-vqvae --vqvae-steps 100000
python pipeline.py train-gpt   --gpt-steps 200000
python pipeline.py train-diffusion --inbetween-steps 200000

# Resume/override
python pipeline.py train --stage inbetween --inbetween-resume checkpoints/composite_inbetween_step100000.pt --inbetween-steps 200000
```

Useful options for `train`:
- `--force` — retrain even if a checkpoint already exists
- `--vqvae-steps`, `--gpt-steps`, `--inbetween-steps`
- `--inbetween-resume`, `--inbetween-ckpt-prefix`
- `--keyframe-source-dir` — use pre-generated ARLM motions as conditioning
- `--disable-selector` — disable the learned keyframe selector
- `--lr`, `--batch-size`, `--grad-accum-steps`
- `--scheduler-type cosine|constant`, `--warmup-ratio`, `--min-lr-ratio`
- `--inbetween-lr`, `--selector-lr`, `--ema-decay`
- `--val-interval`, `--val-batches`

### Full Sample Generation

```bash
python pipeline.py sample --prompt "a person walks forward" --length 120 --out-dir samples --out-name sample_a
```

Useful options:
- `--diff-guidance` — classifier-free guidance scale (default 2.5)
- `--disable-selector` — fall back to heuristic keyframe selection
- `--keyframe-strategy interval|random`, `--keyframe-interval`

### Fine-Tune Diffusion From ARLM Pred Files

Converts `*_pred.npy` files and runs in-between fine-tuning:

```bash
python pipeline.py finetune --sample-dir TRAIN_ALL_GEN --resume-ckpt checkpoints/composite_inbetween_step200000.pt --finetune-steps 40000
```

### ARLM Generate + Fine-Tune (All-in-One)

Generates ARLM conditioning motions and launches in-between fine-tune:

```bash
python pipeline.py arlm-finetune --t2mgpt-root D:/Projects/T2M-GPT --resume-ckpt checkpoints/composite_inbetween_step200000.pt --finetune-total-steps 240000
```

### Retrain + Fine-Tune In One Command

Retrain diffusion in-betweening to 200k with selector enabled, then fine-tune for 40k at reduced LR:

```bash
python pipeline.py retrain-finetune --base-steps 200000 --finetune-steps 40000 --base-lr 1e-4 --finetune-lr 1e-5 --batch-size 32 --grad-accum-steps 4
```

Default selector is enabled.  This gives an effective batch size of 128 via gradient accumulation.

### Evaluate

Run HumanML3D metrics (FID, R-Precision, Diversity, Multimodality):

```bash
python pipeline.py evaluate --models composite --num-samples 256
python pipeline.py evaluate --models all --composite-gpt-steps 200000 --composite-inbetween-steps 200000
```

### Visualize

```bash
python pipeline.py visualize samples/sample_a_motion.npy
```

## Recommended Training Recipe

```bash
# 1. Train VQ-VAE (100k steps)
python pipeline.py train-vqvae --vqvae-steps 100000 --batch-size 32 --grad-accum-steps 1

# 2. Train GPT on VQ-VAE tokens (200k steps)
python pipeline.py train-gpt --gpt-steps 200000 --batch-size 32 --grad-accum-steps 4

# 3. Train diffusion in-betweening (200k steps)
python pipeline.py train-diffusion --inbetween-steps 200000 --batch-size 32 --grad-accum-steps 4

# 4. Generate ARLM motions for the training split and fine-tune inbetweening on them
python pipeline.py arlm-finetune \
    --t2mgpt-root D:/Projects/T2M-GPT \
    --resume-ckpt checkpoints/composite_inbetween_best.pt \
    --finetune-total-steps 240000 \
    --finetune-lr 1e-5

# 5. Evaluate
python pipeline.py evaluate --models composite --num-samples 1024
```

## What Gets Saved

### Checkpoints

- `checkpoints/composite_vqvae_stepN.pt`
- `checkpoints/composite_gpt_stepN.pt`
- `checkpoints/composite_inbetween_stepN.pt` / `composite_inbetween_best.pt`

In-between checkpoints contain `inbetween`, `inbetween_ema`, and (when selector training is on) `selector` + `selector_ema`.

### Training Diagnostics

- `training_logs/vqvae_YYYYMMDD_HHMMSS.{csv,png}`
- `training_logs/gpt_YYYYMMDD_HHMMSS.{csv,png}`
- `training_logs/inbetween_YYYYMMDD_HHMMSS.{csv,png}`

### Sample Outputs

- `samples/<name>_motion.npy`
- `samples/<name>_keyframes.npy`
- `samples/<name>_keyframe_indices.npy`
- `samples/<name>_meta.json`
- `samples/<name>.gif` (or HTML fallback)

## Main Files

| File | Purpose |
|---|---|
| `pipeline.py` | Unified CLI entry point |
| `config.py` | All training/inference defaults |
| `train.py` | Core training loop (VQ-VAE → GPT → Diffusion) |
| `run_full_sample.py` | End-to-end sample generation |
| `generate.py` | Inference helpers (`load_models`, `generate_composite`) |
| `finetune_inbetween_on_arlm_samples.py` | Convert ARLM outputs and fine-tune diffusion |
| `arlm_generate_and_finetune.py` | ARLM generation + fine-tune orchestration |
| `eval_ML3D.py` | HumanML3D evaluation (FID, R-Precision, …) |
| `visualize.py` | Motion rendering utilities |
| `models/vqvae.py` | ResBlock VQ-VAE |
| `models/gpt.py` | Cross-attention GPT |
| `models/diffusion.py` | Diffusion in-betweening + keyframe selector |

## Environment Notes

- Use the project virtual environment for consistent dependencies.
- For T2M-GPT integration, `--t2mgpt-root` must point to a valid T2M-GPT checkout with pretrained weights.

