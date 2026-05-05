from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


SELECTOR_MODE_CHOICES = (
    'reconstruction',
    'random',
    'interval',
    'energy',
    'pose_extrema',
    'interpolation_error',
    'contact_transition',
)


def _apply_endpoint_constraints(
    probs: torch.Tensor,
    hard: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    endpoint_mask = torch.zeros_like(hard)
    has_valid = valid_mask.any(dim=1)
    if has_valid.any():
        batch_idx = torch.arange(valid_mask.shape[0], device=valid_mask.device)[has_valid]
        last_idx = valid_mask.long().sum(dim=1).clamp(min=1) - 1
        endpoint_mask[batch_idx, 0] = 1.0
        endpoint_mask[batch_idx, last_idx[batch_idx]] = 1.0

    probs = torch.maximum(probs, endpoint_mask)
    hard = torch.maximum(hard, endpoint_mask)
    probs = probs * valid_mask.float()
    hard = hard * valid_mask.float()
    return probs, hard


def _topk_straight_through(
    probs: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratio: float = 0.2,
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hard = torch.zeros_like(probs)
    probs, _ = _apply_endpoint_constraints(probs, hard, valid_mask)

    for b in range(probs.shape[0]):
        valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() <= 2:
            hard[b, valid_idx] = 1.0
            continue

        budget = max(2, resolve_keyframe_budget(int(valid_idx.numel()), budget_ratio, topk))
        inner_idx = valid_idx[1:-1]
        inner_budget = max(0, budget - 2)

        hard[b, valid_idx[0]] = 1.0
        hard[b, valid_idx[-1]] = 1.0
        if inner_budget > 0 and inner_idx.numel() > 0:
            inner_scores = probs[b, inner_idx]
            chosen = inner_idx[torch.topk(inner_scores, k=min(inner_budget, inner_idx.numel())).indices]
            hard[b, chosen] = 1.0

    probs, hard = _apply_endpoint_constraints(probs, hard, valid_mask)
    st_mask = hard + probs - probs.detach()
    st_mask = st_mask * valid_mask.float()
    return probs, st_mask


def resolve_keyframe_budget(
    length: int,
    budget_ratio: float = 0.2,
    topk: Optional[int] = None,
) -> int:
    if length <= 0:
        return 0
    if topk is None:
        budget = int(round(length * float(budget_ratio)))
    else:
        budget = int(topk)
    return max(1, min(length, budget))


class BaseKeyframeSelector(nn.Module):
    is_trainable: bool = True

    def __init__(self, threshold: float = 0.5, topk: Optional[int] = None, budget_ratio: Optional[float] = None):
        super().__init__()
        self.threshold = threshold
        self.topk = topk
        self.budget_ratio = budget_ratio

    def compute_auxiliary_loss(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor],
        probs: torch.Tensor,
        st_mask: torch.Tensor,
    ) -> torch.Tensor:
        del motion, valid_mask, cond, st_mask
        return torch.zeros((), device=probs.device)
