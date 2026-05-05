from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .common import BaseKeyframeSelector, _topk_straight_through
from .motion_features import normalize_scores, root_relative_joints


def _local_interpolation_error(features: torch.Tensor) -> torch.Tensor:
    if features.shape[1] <= 2:
        return features.new_zeros(features.shape[0], features.shape[1])
    prev_frame = features[:, :-2]
    next_frame = features[:, 2:]
    midpoint = 0.5 * (prev_frame + next_frame)
    center = features[:, 1:-1]
    error = (center - midpoint).flatten(start_dim=2).norm(dim=-1)
    return F.pad(error, (1, 1))


class InterpolationErrorSelector(BaseKeyframeSelector):
    """Select top-k frames with the largest local interpolation error."""

    is_trainable = False

    def __init__(
        self,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        budget_ratio: float = 0.2,
        **_: object,
    ):
        super().__init__(threshold=threshold, topk=topk, budget_ratio=budget_ratio)

    def _joint_space_error(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        score = _local_interpolation_error(root_relative_joints(motion))
        return normalize_scores(score, valid_mask)

    def _fallback_feature_error(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        score = _local_interpolation_error(motion)
        return normalize_scores(score, valid_mask)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cond
        if motion.shape[-1] >= 67:
            probs = self._joint_space_error(motion, valid_mask)
        else:
            probs = self._fallback_feature_error(motion, valid_mask)
        probs = probs.clamp(0.0, 1.0) * valid_mask.float()
        return _topk_straight_through(
            probs,
            valid_mask,
            budget_ratio=self.budget_ratio or 0.2,
            topk=self.topk,
        )
