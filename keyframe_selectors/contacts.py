from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .common import BaseKeyframeSelector, _topk_straight_through
from .motion_features import normalize_scores, recover_joints_from_ric


class ContactTransitionSelector(BaseKeyframeSelector):
    """Select top-k frames around foot contact state changes."""

    is_trainable = False
    FOOT_JOINTS = (7, 8, 10, 11)

    def __init__(
        self,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        budget_ratio: float = 0.2,
        **_: object,
    ):
        super().__init__(threshold=threshold, topk=topk, budget_ratio=budget_ratio)

    def _joint_space_contact_transitions(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        joints = recover_joints_from_ric(motion)
        feet = joints[:, :, list(self.FOOT_JOINTS), :]
        foot_height = feet[..., 1]
        foot_speed = torch.diff(feet, dim=1, prepend=feet[:, :1]).norm(dim=-1)

        scores = motion.new_zeros(motion.shape[0], motion.shape[1])
        for b in range(motion.shape[0]):
            valid = valid_mask[b]
            if valid.sum() <= 1:
                continue

            valid_height = foot_height[b, valid].flatten()
            valid_speed = foot_speed[b, valid].flatten()
            height_cut = torch.quantile(valid_height, 0.25)
            speed_cut = torch.quantile(valid_speed, 0.25)

            contact = (foot_height[b] <= height_cut) & (foot_speed[b] <= speed_cut)
            transition = torch.diff(contact.float(), dim=0, prepend=contact[:1].float()).abs().mean(dim=-1)
            transition = torch.maximum(transition, F.pad(transition[1:], (0, 1)))
            scores[b] = transition

        return normalize_scores(scores, valid_mask)

    def _fallback_contact_transitions(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return motion.new_zeros(motion.shape[0], motion.shape[1]) * valid_mask.float()

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cond
        if motion.shape[-1] >= 67:
            probs = self._joint_space_contact_transitions(motion, valid_mask)
        else:
            probs = self._fallback_contact_transitions(motion, valid_mask)
        probs = probs.clamp(0.0, 1.0) * valid_mask.float()
        return _topk_straight_through(
            probs,
            valid_mask,
            budget_ratio=self.budget_ratio or 0.2,
            topk=self.topk,
        )
