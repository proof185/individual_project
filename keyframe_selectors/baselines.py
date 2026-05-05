from __future__ import annotations

from typing import Optional, Tuple

import torch

from .common import BaseKeyframeSelector, resolve_keyframe_budget


def _baseline_budget(length: int, budget_ratio: float, topk: Optional[int]) -> int:
    budget = resolve_keyframe_budget(length, budget_ratio, topk)
    if length >= 2:
        budget = max(2, budget)
    return budget


def _mask_from_indices(
    valid_mask: torch.Tensor,
    indices_by_batch: list[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.zeros_like(valid_mask, dtype=torch.float32)
    hard = torch.zeros_like(probs)
    for batch_idx, indices in enumerate(indices_by_batch):
        if indices.numel() == 0:
            continue
        indices = indices.to(valid_mask.device).long()
        indices = indices[(indices >= 0) & (indices < valid_mask.shape[1])]
        indices = indices[valid_mask[batch_idx, indices]]
        if indices.numel() == 0:
            continue
        probs[batch_idx, indices] = 1.0
        hard[batch_idx, indices] = 1.0
    return probs, hard


class RandomKeyframeSelector(BaseKeyframeSelector):
    """Select a random fixed-budget set of keyframes."""

    is_trainable = False

    def __init__(
        self,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        budget_ratio: float = 0.2,
        **_: object,
    ):
        super().__init__(threshold=threshold, topk=topk, budget_ratio=budget_ratio)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del motion, cond
        indices_by_batch = []
        for b in range(valid_mask.shape[0]):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
            budget = _baseline_budget(int(valid_idx.numel()), self.budget_ratio or 0.2, self.topk)
            if budget == 0:
                indices_by_batch.append(valid_idx[:0])
                continue
            if valid_idx.numel() <= 2:
                indices_by_batch.append(valid_idx)
                continue
            inner_idx = valid_idx[1:-1]
            inner_budget = max(0, budget - 2)
            if inner_budget > 0:
                perm = torch.randperm(inner_idx.numel(), device=inner_idx.device)[:inner_budget]
                chosen = torch.cat([valid_idx[:1], inner_idx[perm], valid_idx[-1:]])
            else:
                chosen = torch.cat([valid_idx[:1], valid_idx[-1:]])
            indices_by_batch.append(chosen.sort().values)
        return _mask_from_indices(valid_mask, indices_by_batch)


class IntervalKeyframeSelector(BaseKeyframeSelector):
    """Select an approximately uniform fixed-budget set of keyframes."""

    is_trainable = False

    def __init__(
        self,
        threshold: float = 0.5,
        topk: Optional[int] = None,
        budget_ratio: float = 0.2,
        **_: object,
    ):
        super().__init__(threshold=threshold, topk=topk, budget_ratio=budget_ratio)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del motion, cond
        indices_by_batch = []
        for b in range(valid_mask.shape[0]):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
            budget = _baseline_budget(int(valid_idx.numel()), self.budget_ratio or 0.2, self.topk)
            if budget == 0:
                indices_by_batch.append(valid_idx[:0])
                continue
            if budget == 1:
                indices_by_batch.append(valid_idx[:1])
                continue
            positions = torch.linspace(0, valid_idx.numel() - 1, steps=budget, device=valid_idx.device)
            chosen = valid_idx[positions.round().long().unique(sorted=True)]
            indices_by_batch.append(chosen)
        return _mask_from_indices(valid_mask, indices_by_batch)
