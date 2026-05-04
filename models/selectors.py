from typing import Optional

from .selector_modules import (
    BaseKeyframeSelector,
    InformationGainKeyframeSelector,
    MotionSaliencySelector,
    RetrievalGainKeyframeSelector,
    SELECTOR_MODE_CHOICES,
    TextAlignmentKeyframeSelector,
)


KeyframeSelector = TextAlignmentKeyframeSelector


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
    if mode == 'information_gain':
        return InformationGainKeyframeSelector(
            feature_dim=feature_dim,
            cond_dim=cond_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            threshold=threshold,
            budget_ratio=(0.1 if budget_ratio is None else budget_ratio),
        )
    if mode == 'retrieval_gain':
        return RetrievalGainKeyframeSelector(
            feature_dim=feature_dim,
            cond_dim=cond_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            threshold=threshold,
            budget_ratio=(0.1 if budget_ratio is None else budget_ratio),
        )
    if mode in {'text_alignment', 'r_precision', 'transformer'}:
        return TextAlignmentKeyframeSelector(
            feature_dim=feature_dim,
            cond_dim=cond_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            threshold=threshold,
        )
    if mode == 'saliency':
        return MotionSaliencySelector(threshold=threshold)
    raise ValueError(f'Unknown selector mode {mode!r}. Expected one of: {SELECTOR_MODE_CHOICES}.')