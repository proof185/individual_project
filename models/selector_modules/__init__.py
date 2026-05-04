from typing import Optional

from .common import BaseKeyframeSelector, SELECTOR_MODE_CHOICES
from .reconstruction import ReconstructionKeyframeSelector
from .saliency import MotionSaliencySelector

KeyframeSelector = ReconstructionKeyframeSelector


def build_keyframe_selector(
    mode: str,
    feature_dim: int,
    cond_dim: int = 512,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    dropout: float = 0.1,
    max_len: int = 256,
    threshold: float = 0.5,
    topk: Optional[int] = None,
    budget_ratio: Optional[float] = None,
) -> BaseKeyframeSelector:
    mode = mode.lower().strip()
    if mode == 'reconstruction':
        return ReconstructionKeyframeSelector(
            feature_dim=feature_dim,
            cond_dim=cond_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            threshold=threshold,
            budget_ratio=(0.2 if budget_ratio is None else budget_ratio),
        )
    if mode == 'saliency':
        return MotionSaliencySelector(threshold=threshold)
    raise ValueError(f'Unknown selector mode {mode!r}. Expected one of: {SELECTOR_MODE_CHOICES}.')


__all__ = [
    'BaseKeyframeSelector',
    'KeyframeSelector',
    'SELECTOR_MODE_CHOICES',
    'ReconstructionKeyframeSelector',
    'MotionSaliencySelector',
    'build_keyframe_selector',
]
