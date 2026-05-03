"""
Models package for composite motion generation.

Contains:
- vqvae: Vector quantized variational autoencoder for motion tokenization
- gpt: Autoregressive transformer for keyframe generation
- diffusion: Diffusion models for motion in-betweening
"""

from .vqvae import VectorQuantizer, MotionVQVAE
from .gpt import CausalSelfAttention, GPTBlock, MotionGPT
from .diffusion import (
    cosine_beta_schedule,
    timestep_embedding,
    InbetweenDiffusion,
    InbetweenTransformer,
)
from .selectors import SELECTOR_MODE_CHOICES, KeyframeSelector, build_keyframe_selector

__all__ = [
    'VectorQuantizer',
    'MotionVQVAE',
    'CausalSelfAttention',
    'GPTBlock',
    'MotionGPT',
    'cosine_beta_schedule',
    'timestep_embedding',
    'InbetweenDiffusion',
    'InbetweenTransformer',
    'SELECTOR_MODE_CHOICES',
    'KeyframeSelector',
    'build_keyframe_selector',
]
