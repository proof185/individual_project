from typing import Optional

from .baselines import IntervalKeyframeSelector, RandomKeyframeSelector
from .common import BaseKeyframeSelector, SELECTOR_MODE_CHOICES
from .contacts import ContactTransitionSelector
from .dynamics import MotionEnergySelector, PoseExtremaSelector
from .inbetweening import InterpolationErrorSelector
from .reconstruction import ReconstructionKeyframeSelector

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
    if mode == 'random':
        return RandomKeyframeSelector(
            threshold=threshold,
            topk=topk,
            budget_ratio=(0.2 if budget_ratio is None else budget_ratio),
        )
    if mode == 'interval':
        return IntervalKeyframeSelector(
            threshold=threshold,
            topk=topk,
            budget_ratio=(0.2 if budget_ratio is None else budget_ratio),
        )
    if mode == 'energy':
        return MotionEnergySelector(
            threshold=threshold,
            topk=topk,
            budget_ratio=(0.2 if budget_ratio is None else budget_ratio),
        )
    if mode == 'pose_extrema':
        return PoseExtremaSelector(
            threshold=threshold,
            topk=topk,
            budget_ratio=(0.2 if budget_ratio is None else budget_ratio),
        )
    if mode == 'interpolation_error':
        return InterpolationErrorSelector(
            threshold=threshold,
            topk=topk,
            budget_ratio=(0.2 if budget_ratio is None else budget_ratio),
        )
    if mode == 'contact_transition':
        return ContactTransitionSelector(
            threshold=threshold,
            topk=topk,
            budget_ratio=(0.2 if budget_ratio is None else budget_ratio),
        )
    raise ValueError(f'Unknown selector mode {mode!r}. Expected one of: {SELECTOR_MODE_CHOICES}.')


__all__ = [
    'BaseKeyframeSelector',
    'KeyframeSelector',
    'SELECTOR_MODE_CHOICES',
    'ReconstructionKeyframeSelector',
    'RandomKeyframeSelector',
    'IntervalKeyframeSelector',
    'MotionEnergySelector',
    'PoseExtremaSelector',
    'InterpolationErrorSelector',
    'ContactTransitionSelector',
    'build_keyframe_selector',
]
