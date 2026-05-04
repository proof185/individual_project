from __future__ import annotations

from typing import Optional, Tuple

import torch

from .common import (
    BaseKeyframeSelector,
    _adaptive_saliency_straight_through,
    _diff_norm,
    _group_motion_energy,
    _normalize_scores,
    _recover_joints_from_ric,
    _root_turning_score,
)


class MotionSaliencySelector(BaseKeyframeSelector):
    is_trainable = False

    ROOT_AND_SPINE = (0, 3, 6, 9, 12, 15)
    FEET = (7, 8, 10, 11)
    HANDS = (13, 14, 16, 17, 18, 19, 20, 21)
    OTHER = (1, 2, 4, 5)

    def __init__(self, threshold: float = 0.5, **_: object):
        super().__init__(threshold=threshold)

    def _joint_space_saliency(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        joints = _recover_joints_from_ric(motion)
        root_spine = _group_motion_energy(joints, self.ROOT_AND_SPINE)
        feet = _group_motion_energy(joints, self.FEET)
        hands = _group_motion_energy(joints, self.HANDS)
        other = _group_motion_energy(joints, self.OTHER)
        turning = _root_turning_score(joints)

        saliency = (
            1.6 * root_spine
            + 1.4 * feet
            + 1.2 * hands
            + 0.8 * other
            + 1.6 * turning
        )
        saliency = saliency * valid_mask.float()
        saliency = _normalize_scores(saliency, valid_mask)
        local_max = torch.nn.functional.max_pool1d(saliency.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        peak_mask = (saliency >= (local_max - 1e-6)).float()
        return (saliency * peak_mask) * valid_mask.float()

    def _fallback_feature_saliency(self, motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        velocity = _diff_norm(motion, order=1)
        acceleration = _diff_norm(motion, order=2)
        jerk = _diff_norm(motion, order=3)
        return _normalize_scores(velocity + 0.5 * acceleration + 0.25 * jerk, valid_mask)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cond
        if motion.shape[-1] >= 67:
            probs = self._joint_space_saliency(motion, valid_mask)
        else:
            probs = self._fallback_feature_saliency(motion, valid_mask)
        probs = probs.clamp(0.0, 1.0) * valid_mask.float()
        return _adaptive_saliency_straight_through(probs, valid_mask, self.threshold)