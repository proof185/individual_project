from __future__ import annotations

from typing import Optional, Tuple

import torch

from .common import (
    BaseKeyframeSelector,
    _topk_straight_through,
)
from .motion_features import normalize_scores, root_relative_joints, velocity_norm


class MotionEnergySelector(BaseKeyframeSelector):
    """Select top-k frames with the largest root-relative joint velocity."""

    is_trainable = False

    def __init__(
        self,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        budget_ratio: float = 0.2,
        **_: object,
    ):
        super().__init__(threshold=threshold, topk=topk, budget_ratio=budget_ratio)

    def _joint_space_energy(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        root_relative = root_relative_joints(motion)
        velocity = torch.diff(root_relative, dim=1, prepend=root_relative[:, :1])
        energy = velocity.norm(dim=-1).mean(dim=-1)
        return normalize_scores(energy, valid_mask)

    def _fallback_feature_energy(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return normalize_scores(velocity_norm(motion), valid_mask)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cond
        if motion.shape[-1] >= 67:
            probs = self._joint_space_energy(motion, valid_mask)
        else:
            probs = self._fallback_feature_energy(motion, valid_mask)
        probs = probs.clamp(0.0, 1.0) * valid_mask.float()
        return _topk_straight_through(
            probs,
            valid_mask,
            budget_ratio=self.budget_ratio or 0.2,
            topk=self.topk,
        )


class PoseExtremaSelector(BaseKeyframeSelector):
    """Select top-k frames farthest from a neutral/root-relative pose."""

    is_trainable = False

    def __init__(
        self,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        budget_ratio: float = 0.2,
        **_: object,
    ):
        super().__init__(threshold=threshold, topk=topk, budget_ratio=budget_ratio)

    def _joint_space_pose_extrema(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        root_relative = root_relative_joints(motion)
        neutral = (
            root_relative * valid_mask.float().view(valid_mask.shape[0], valid_mask.shape[1], 1, 1)
        ).sum(dim=1, keepdim=True) / valid_mask.float().sum(dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
        score = (root_relative - neutral).norm(dim=-1).mean(dim=-1)
        return normalize_scores(score, valid_mask)

    def _fallback_feature_pose_extrema(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.float().unsqueeze(-1)
        neutral = (motion * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return normalize_scores((motion - neutral).norm(dim=-1), valid_mask)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cond
        if motion.shape[-1] >= 67:
            probs = self._joint_space_pose_extrema(motion, valid_mask)
        else:
            probs = self._fallback_feature_pose_extrema(motion, valid_mask)
        probs = probs.clamp(0.0, 1.0) * valid_mask.float()
        return _topk_straight_through(
            probs,
            valid_mask,
            budget_ratio=self.budget_ratio or 0.2,
            topk=self.topk,
        )
