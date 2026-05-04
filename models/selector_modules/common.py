from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SELECTOR_MODE_CHOICES = (
    'information_gain',
    'retrieval_gain',
    'text_alignment',
    'saliency',
)


def _masked_mean(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.float().unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _normalize_scores(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    scores = scores * valid_mask.float()
    max_score = scores.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return (scores / max_score).clamp(0.0, 1.0) * valid_mask.float()


def _diff_norm(motion: torch.Tensor, order: int) -> torch.Tensor:
    out = motion
    for _ in range(order):
        out = out[:, 1:, :] - out[:, :-1, :]
    if out.shape[1] == 0:
        return motion.new_zeros(motion.shape[0], motion.shape[1])
    score = out.norm(dim=-1)
    if order > 0:
        score = F.pad(score, (order, 0))
    return score[:, : motion.shape[1]]


def _local_interpolation_error(motion: torch.Tensor) -> torch.Tensor:
    if motion.shape[1] <= 2:
        return motion.new_zeros(motion.shape[0], motion.shape[1])
    prev = motion[:, :-2, :]
    nxt = motion[:, 2:, :]
    midpoint = 0.5 * (prev + nxt)
    center = motion[:, 1:-1, :]
    err = (center - midpoint).norm(dim=-1)
    return F.pad(err, (1, 1))


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


def _straight_through(
    probs: torch.Tensor,
    threshold: float,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hard = (probs > threshold).float()
    probs, hard = _apply_endpoint_constraints(probs, hard, valid_mask)
    st_mask = hard + probs - probs.detach()
    st_mask = st_mask * valid_mask.float()
    return probs, st_mask


def _adaptive_saliency_straight_through(
    probs: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hard = torch.zeros_like(probs)
    probs, _ = _apply_endpoint_constraints(probs, hard, valid_mask)

    for b in range(probs.shape[0]):
        valid_scores = probs[b][valid_mask[b]]
        if valid_scores.numel() == 0:
            continue
        if valid_scores.numel() <= 2:
            hard[b][valid_mask[b]] = 1.0
            continue

        activity = valid_scores.mean()
        base_quantile = 0.85 + 0.10 * float(threshold)
        quantile = base_quantile - 0.10 * float(activity)
        quantile = max(0.80, min(0.95, quantile))
        cut = torch.quantile(valid_scores, quantile)
        hard[b] = (probs[b] >= cut).float()

    probs, hard = _apply_endpoint_constraints(probs, hard, valid_mask)
    st_mask = hard + probs - probs.detach()
    st_mask = st_mask * valid_mask.float()
    return probs, st_mask


def _budget_topk_straight_through(
    probs: torch.Tensor,
    valid_mask: torch.Tensor,
    budget_ratio: float,
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

        budget = max(2, min(int(valid_idx.numel()), int(round(valid_idx.numel() * float(budget_ratio)))))
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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    original_shape = v.shape
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2.0 * (q[:, :1] * uv + uuv)).view(original_shape)


def _recover_root_rot_pos(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), dtype=data.dtype, device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), dtype=data.dtype, device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def _recover_joints_from_ric(data: torch.Tensor, joints_num: int = 22) -> torch.Tensor:
    r_rot_quat, r_pos = _recover_root_rot_pos(data)
    joints = data[..., 4:(joints_num - 1) * 3 + 4]
    joints = joints.view(joints.shape[:-1] + (joints_num - 1, 3))
    joints = _qrot(
        _qinv(r_rot_quat[..., None, :]).expand(joints.shape[:-1] + (4,)),
        joints,
    )
    joints[..., 0] += r_pos[..., 0:1]
    joints[..., 2] += r_pos[..., 2:3]
    return torch.cat([r_pos.unsqueeze(-2), joints], dim=-2)


def _group_motion_energy(joints: torch.Tensor, joint_indices: Tuple[int, ...]) -> torch.Tensor:
    group = joints[:, :, list(joint_indices), :]
    velocity = torch.diff(group, dim=1, prepend=group[:, :1])
    acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1])
    jerk = torch.diff(acceleration, dim=1, prepend=acceleration[:, :1])
    vel_score = velocity.norm(dim=-1).mean(dim=-1)
    acc_score = acceleration.norm(dim=-1).mean(dim=-1)
    jerk_score = jerk.norm(dim=-1).mean(dim=-1)
    return vel_score + 0.6 * acc_score + 0.35 * jerk_score


def _root_turning_score(joints: torch.Tensor) -> torch.Tensor:
    root = joints[:, :, 0, :]
    planar_vel = root[:, 1:, [0, 2]] - root[:, :-1, [0, 2]]
    planar_vel = F.pad(planar_vel, (0, 0, 1, 0))
    heading = torch.atan2(planar_vel[..., 1], planar_vel[..., 0])
    heading_delta = torch.diff(heading, dim=1, prepend=heading[:, :1])
    heading_delta = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta)).abs()
    speed = planar_vel.norm(dim=-1)
    return heading_delta * speed


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
        oracle_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del motion, valid_mask, cond, st_mask, oracle_target
        return torch.zeros((), device=probs.device)