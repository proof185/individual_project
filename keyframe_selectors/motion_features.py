from __future__ import annotations

from typing import Tuple

import torch


def normalize_scores(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    scores = scores * valid_mask.float()
    max_score = scores.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return (scores / max_score).clamp(0.0, 1.0) * valid_mask.float()


def velocity_norm(motion: torch.Tensor) -> torch.Tensor:
    velocity = torch.diff(motion, dim=1, prepend=motion[:, :1])
    return velocity.norm(dim=-1)


def qinv(q: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    original_shape = v.shape
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2.0 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), dtype=data.dtype, device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), dtype=data.dtype, device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_joints_from_ric(data: torch.Tensor, joints_num: int = 22) -> torch.Tensor:
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    joints = data[..., 4:(joints_num - 1) * 3 + 4]
    joints = joints.view(joints.shape[:-1] + (joints_num - 1, 3))
    joints = qrot(
        qinv(r_rot_quat[..., None, :]).expand(joints.shape[:-1] + (4,)),
        joints,
    )
    joints[..., 0] += r_pos[..., 0:1]
    joints[..., 2] += r_pos[..., 2:3]
    return torch.cat([r_pos.unsqueeze(-2), joints], dim=-2)


def root_relative_joints(motion: torch.Tensor) -> torch.Tensor:
    joints = recover_joints_from_ric(motion)
    return joints - joints[:, :, :1, :]
