"""Configuration for composite motion generation."""

from dataclasses import dataclass


@dataclass
class InbetweenTrainConfig:
    # Data
    root: str = 'humanml'
    max_len: int = 196
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    keyframe_source_dir: str | None = None

    # Optimisation
    lr: float = 1e-4
    inbetween_steps: int = 75_000
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.05
    inbetween_lr: float | None = None
    selector_lr: float | None = None
    selector_lr_scale: float = 0.5
    ema_decay: float = 0.999
    use_ema_for_sampling: bool = True

    # Validation
    val_split: str = 'val'
    val_batch_size: int = 32
    val_interval: int = 1000
    val_batches: int = 10

    # Conditioning/keyframes
    keyframe_interval: int = 5
    keyframe_strategy: str = 'random'
    keyframe_count: int | None = None
    keyframe_min: int = 8
    keyframe_max: int = 28
    keyframe_include_ends: bool = True

    # Loss weights
    inbetween_velocity_weight: float = 0.75
    boundary_velocity_weight: float = 1.0
    boundary_acceleration_weight: float = 0.25
    boundary_jerk_weight: float = 0.1
    transition_consistency_weight: float = 0.75
    transition_window: int = 4
    transition_sigma: float = 1.5
    transition_velocity_weight: float = 1.0
    transition_acceleration_weight: float = 0.5

    # Selector
    use_learned_keyframe_selector: bool = False
    selector_mode: str = 'text_alignment'
    selector_d_model: int = 256
    selector_layers: int = 4
    selector_heads: int = 4
    selector_dropout: float = 0.1
    selector_threshold: float = 0.5
    selector_target_ratio: float = 0.2
    selector_budget_weight: float = 5.0
    selector_entropy_weight: float = 0.02
    selector_aux_weight: float = 0.5
    selector_curriculum_fraction: float = 0.3
    freeze_inbetween_for_selector: bool = False
    selector_oracle_ckpt_path: str | None = None
    external_inbetween_ckpt_path: str | None = None
    selector_oracle_timesteps: int = 3
    selector_eval_root: str | None = None
    selector_retrieval_negatives: int = 31

    # Diffusion/model
    T_diffusion: int = 1000
    p_uncond: float = 0.1
    dropout: float = 0.1
    inbetween_d_model: int = 512
    inbetween_layers: int = 12
    inbetween_heads: int = 8


@dataclass
class SelectorTrainConfig:
    # Data
    root: str = 'humanml'
    max_len: int = 196
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Optimization
    selector_steps: int = 75_000
    selector_lr: float = 5e-5
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.05
    ema_decay: float = 0.999
    use_ema_for_sampling: bool = True

    # Validation
    val_split: str = 'val'
    val_batch_size: int = 32
    val_interval: int = 1000
    val_batches: int = 10

    # Keyframes
    keyframe_strategy: str = 'random'

    # Selector
    selector_mode: str = 'text_alignment'
    selector_d_model: int = 256
    selector_layers: int = 4
    selector_heads: int = 4
    selector_dropout: float = 0.1
    selector_threshold: float = 0.5
    selector_target_ratio: float = 0.2
    selector_budget_weight: float = 5.0
    selector_entropy_weight: float = 0.02
    selector_aux_weight: float = 0.5
    selector_curriculum_fraction: float = 0.3
    selector_oracle_ckpt_path: str | None = None
    external_inbetween_ckpt_path: str = 'D:\Projects\diffusion-motion-inbetweening\save\condmdi_randomframes\model000750000.pt'
    selector_oracle_timesteps: int = 3
    selector_eval_root: str | None = None
    selector_retrieval_negatives: int = 31

    # Conditioning
    p_uncond: float = 0.1


@dataclass
class CompositeConfig:
    root: str = 'humanml'
    split: str = 'train'
    max_len: int = 196
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    keyframe_source_dir: str | None = None

    lr: float = 1e-4
    vqvae_steps: int = 100_000   # Stage 1a: VQ-VAE training (more steps for larger model)
    gpt_steps: int = 200_000     # Stage 1b: GPT training (more steps for larger model)
    inbetween_steps: int = 75_000  # Stage 2: Diffusion in-betweening
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # Optimizer/LR schedule
    scheduler_type: str = 'cosine'  # 'cosine' or 'constant'
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.05

    # In-between optimizer split LRs
    inbetween_lr: float | None = None
    selector_lr: float | None = None
    selector_lr_scale: float = 0.5

    # EMA for in-between model/selector
    ema_decay: float = 0.999
    use_ema_for_sampling: bool = True

    # In-between validation / checkpoint selection
    val_split: str = 'val'
    val_batch_size: int = 32
    val_interval: int = 1000
    val_batches: int = 10

    # In-betweening config
    keyframe_interval: int = 5   # Generate keyframe every n frames
    keyframe_strategy: str = 'random'  # 'interval' or 'random'
    keyframe_count: int | None = None  # If set, number of random keyframes
    keyframe_min: int = 8  # Min random keyframes when keyframe_count is None
    keyframe_max: int = 28  # Max random keyframes when keyframe_count is None
    keyframe_include_ends: bool = True  # Always include first/last frame
    inbetween_velocity_weight: float = 0.75
    boundary_velocity_weight: float = 1.0
    boundary_acceleration_weight: float = 0.25
    boundary_jerk_weight: float = 0.1
    transition_consistency_weight: float = 0.75
    transition_window: int = 4
    transition_sigma: float = 1.5
    transition_velocity_weight: float = 1.0
    transition_acceleration_weight: float = 0.5

    # Learnable keyframe selector (upstream mask generator)
    use_learned_keyframe_selector: bool = True
    selector_mode: str = 'text_alignment'
    selector_d_model: int = 256
    selector_layers: int = 4
    selector_heads: int = 4
    selector_dropout: float = 0.1
    selector_threshold: float = 0.5
    selector_target_ratio: float = 0.2
    selector_budget_weight: float = 5.0
    selector_entropy_weight: float = 0.02
    selector_aux_weight: float = 0.5
    selector_curriculum_fraction: float = 0.3
    selector_oracle_ckpt_path: str | None = None
    external_inbetween_ckpt_path: str | None = None
    selector_oracle_timesteps: int = 3
    selector_eval_root: str | None = None
    selector_retrieval_negatives: int = 31

    # VQ-VAE
    codebook_size: int = 512   # Larger codebook for richer motion vocabulary
    codebook_dim: int = 256
    commitment_cost: float = 0.25
    downsample_rate: int = 4

    # Diffusion
    T_diffusion: int = 1000

    # Classifier-free guidance
    p_uncond: float = 0.1
    guidance_scale: float = 5.0

    # Model dimensions (GPT)
    d_model: int = 512          # Larger GPT for better token sequence modelling
    n_layers: int = 12          # Deeper GPT
    n_heads: int = 8
    dropout: float = 0.1

    # In-betweening model dimensions
    inbetween_d_model: int = 512
    inbetween_layers: int = 12
    inbetween_heads: int = 8
