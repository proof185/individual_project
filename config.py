"""Configuration for composite motion generation."""

from dataclasses import dataclass


@dataclass
class ReconstructionTrainConfig:
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

    # Selector
    selector_mode: str = 'reconstruction'
    selector_d_model: int = 256
    selector_layers: int = 4
    selector_heads: int = 4
    selector_dropout: float = 0.1
    selector_threshold: float = 0.5
    selector_target_ratio: float = 0.2
    selector_budget_weight: float = 5.0
    selector_entropy_weight: float = 0.02
    selector_aux_weight: float = 1.0
    selector_curriculum_fraction: float = 0.3
    external_inbetween_ckpt_path: str = '../diffusion-motion-inbetweening/save/condmdi_randomframes/model000750000.pt'
    selector_reconstruction_timesteps: int = 3
    T_diffusion: int = 1000

    # Conditioning
    p_uncond: float = 0.1


@dataclass
class CompositeConfig:
    root: str = 'humanml'
    max_len: int = 196

    # Heuristic fallback keyframes used when no learned selector is available.
    keyframe_interval: int = 5
    keyframe_strategy: str = 'random'
    keyframe_count: int | None = None
    keyframe_min: int = 8
    keyframe_max: int = 28
    keyframe_include_ends: bool = True

    # Learnable keyframe selector attached from selector checkpoints.
    use_learned_keyframe_selector: bool = True
    selector_mode: str = 'reconstruction'
    selector_d_model: int = 256
    selector_layers: int = 4
    selector_heads: int = 4
    selector_dropout: float = 0.1
    selector_threshold: float = 0.5
    selector_target_ratio: float = 0.2

    # Classifier-free guidance
    guidance_scale: float = 5.0


@dataclass
class EvalConfig:
    # Evaluation set
    num_samples: int = 1000
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 1234
    device: str | None = None

    # Data/model roots
    humanml_root: str = 'humanml'
    t2mgpt_root: str = 'D:/Projects/T2M-GPT'

    # Compare unmodified T2M-GPT against explicit keyframe selection strategies.
    models: str = 'reconstruction, random, interval, energy, pose_extrema, interpolation_error, contact_transition'
    metrics: str = 'fid,diversity,matching_score,r_precision,jerk,foot_skating'
    r_precision_top_k: str = '1,2,3,5'

    # Output
    results_dir: str = 'samples/eval_results'
    results_path: str | None = None
    csv_path: str | None = None
    save_json: bool = False

    # T2M-GPT generation
    categorical_sample: bool = False
    ar_clamp_sigma: float = 0.0
    arlm_vq_ckpt: str | None = None
    arlm_gpt_ckpt: str | None = None

    # CondMDI / selector checkpoints
    composite_inbetween_ckpt: str | None = None
    composite_reconstruction_ckpt: str | None = None
    condmdi_unconditional_ckpt: str = '../diffusion-motion-inbetweening/save/condmdi_uncond/model000500000.pt'
    disable_reconstruction_selector: bool = False

    # Diffusion sampling
    ddim_steps: int = 50

    # Keyframe selection
    fallback_keyframe_strategy: str | None = None
    keyframe_interval: int = 5
    keyframe_count: int | None = None
    keyframe_min: int | None = None
    keyframe_max: int | None = None
    keyframe_topk: int | None = None
    keyframe_budget_ratio: float = 0.2
    keyframe_include_ends: bool = True
    diff_guidance: float = 2.5
