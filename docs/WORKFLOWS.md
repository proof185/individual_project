# Workflows and Recipes

This page provides copy-paste workflows using the unified CLI.

## Standard Training

Run the full training pipeline:

```bash
python pipeline.py train --stage all
```

Run only in-between training:

```bash
python pipeline.py train --stage inbetween --inbetween-steps 200000
```

Resume in-between training:

```bash
python pipeline.py train --stage inbetween --inbetween-resume checkpoints/composite_inbetween_step100000.pt --inbetween-steps 120000
```

## Selector Behavior

Selector is enabled by default.

Disable selector explicitly:

```bash
python pipeline.py train --stage inbetween --disable-selector
```

For sample generation:

```bash
python pipeline.py sample --prompt "a person turns around" --disable-selector
```

## Generate and Visualize a Sample

```bash
python pipeline.py sample --prompt "a person walks forward then jumps" --length 120 --out-dir samples --out-name demo
python pipeline.py visualize samples/demo_motion.npy
```

## Fine-Tuning With Existing ARLM Pred Files

```bash
python pipeline.py finetune \
  --sample-dir TRAIN_ALL_GEN \
  --prepared-dir humanml/arlm_train_pred_joint_vecs \
  --resume-ckpt checkpoints/composite_inbetween_step100000.pt \
  --start-step 100000 \
  --finetune-steps 20000 \
  --ckpt-prefix finetuned_inbetween
```

## ARLM Generation + Fine-Tuning

```bash
python pipeline.py arlm-finetune \
  --t2mgpt-root D:/Projects/T2M-GPT \
  --output-dir humanml/arlm_new_joint_vecs \
  --resume-ckpt checkpoints/composite_inbetween_step100000.pt \
  --finetune-total-steps 120000
```

## One-Command Retrain Then Fine-Tune

This retrains in-betweening to 200k steps first, then fine-tunes for +40k steps at a reduced learning rate:

```bash
python pipeline.py retrain-finetune \\
  --base-steps 200000 \\
  --finetune-steps 40000 \\
  --base-lr 1e-4 \\
  --finetune-lr 1e-5 \
  --batch-size 32 \
  --grad-accum-steps 4
```

Notes:
- Selector is enabled by default and will be saved in checkpoints.
- Resume checkpoint for fine-tuning is auto-derived as checkpoints/<base-ckpt-prefix>_step<base-steps>.pt.
- Effective batch size is 128 from 32 x 4 accumulation.

## Convergence Logs

Training writes logs in training_logs:

- stage_timestamp.csv: numeric metrics by step.
- stage_timestamp.png: convergence plot.

Use these to compare runs and detect instability early.

## Common Pitfalls

1. Wrong Python environment
Use the project venv interpreter to avoid missing packages.

2. Missing selector in older checkpoints
If a checkpoint was trained with selector disabled, selector weights are absent.

3. T2M-GPT root path
Verify --t2mgpt-root points to a valid T2M-GPT repo with pretrained models.
