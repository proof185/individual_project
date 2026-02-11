# Composite Motion Generation

This project implements a composite motion generation system combining autoregressive and diffusion models.

## Project Structure

```
.
├── config.py              # Configuration dataclass
├── dataset.py             # Dataset and data loading utilities
├── utils.py              # Shared utility functions
├── models/
│   ├── __init__.py       # Models package
│   ├── vqvae.py          # VQ-VAE for motion tokenization
│   ├── gpt.py            # Autoregressive GPT for keyframe generation
│   └── diffusion.py      # Diffusion models for in-betweening
├── train.py              # Training script (3 stages)
├── generate.py           # Inference and generation
└── visualize.py          # Visualization utilities
```

## Usage

### Training

Train all three stages (VQ-VAE, GPT, Diffusion):

```bash
python train.py
```

Training runs in 3 stages:
1. **Stage 1a**: VQ-VAE training for motion tokenization
2. **Stage 1b**: GPT training for autoregressive keyframe generation
3. **Stage 2**: Diffusion training for motion in-betweening

Checkpoints are saved in `checkpoints/` directory.

### Generation

Generate motion from text prompts:

```bash
python generate.py
```

Or use programmatically:

```python
from config import CompositeConfig
from generate import load_models, generate_composite

cfg = CompositeConfig()
vqvae, gpt, inbetween_model, diff_inbetween, clip_model, mean, std, Fdim = load_models(cfg)

motion, keyframes, keyframe_idx = generate_composite(
    vqvae, gpt, inbetween_model, diff_inbetween, clip_model,
    mean, std, Fdim, cfg,
    prompt="a person walks forward",
    length=120,
)
```

### Visualization

Visualize generated motion:

```bash
python visualize.py sample_composite_output.npy
```

Or in Jupyter notebooks:

```python
from visualize import visualize_motion_file

anim = visualize_motion_file("sample_composite_output.npy", title="My Motion")
display(anim)
```

## Configuration

Edit `config.py` to adjust hyperparameters:

```python
from config import CompositeConfig

cfg = CompositeConfig(
    batch_size=64,
    lr=2e-4,
    keyframe_interval=5,
    # ... other parameters
)
```

## Key Components

- **VQ-VAE**: Encodes motion into discrete tokens
- **MotionGPT**: Generates sparse keyframes autoregressively
- **Diffusion In-betweening**: Fills in smooth motion between keyframes
- **CLIP**: Provides text conditioning for generation

## Requirements

- PyTorch
- CLIP (OpenAI)
- NumPy
- Matplotlib (for visualization)
