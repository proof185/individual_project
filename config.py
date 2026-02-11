"""Configuration for composite motion generation."""

from dataclasses import dataclass


@dataclass
class CompositeConfig:
    root: str = 'humanml'
    split: str = 'train'
    max_len: int = 196
    batch_size: int = 32
    num_workers: int = 0

    lr: float = 1e-4
    vqvae_steps: int = 100_000   # Stage 1a: VQ-VAE training
    gpt_steps: int = 200_000     # Stage 1b: GPT training
    inbetween_steps: int = 200_000  # Stage 2: Diffusion in-betweening
    grad_clip: float = 1.0

    # In-betweening config
    keyframe_interval: int = 5   # Generate keyframe every n frames
    keyframe_strategy: str = 'interval'  # 'interval' or 'random'
    keyframe_count: int | None = None  # If set, number of random keyframes
    keyframe_min: int = 6  # Min random keyframes when keyframe_count is None
    keyframe_max: int = 20  # Max random keyframes when keyframe_count is None
    keyframe_include_ends: bool = True  # Always include first/last frame

    # VQ-VAE
    codebook_size: int = 512
    codebook_dim: int = 512
    commitment_cost: float = 0.25
    downsample_rate: int = 4

    # Diffusion
    T_diffusion: int = 1000

    # Classifier-free guidance
    p_uncond: float = 0.1
    guidance_scale: float = 2.5

    # Model dimensions
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
