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


class InformationGainKeyframeSelector(BaseKeyframeSelector):
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
        threshold: float = 0.5,
        budget_ratio: float = 0.1,
        **_: object,
    ):
        super().__init__(threshold=threshold, budget_ratio=budget_ratio)
        self.frame_in = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 1)
        nn.init.constant_(self.out.bias, -2.2)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, T, _ = motion.shape
        if T > self.pos_emb.shape[1]:
            raise ValueError(f'Sequence length {T} exceeds selector max_len {self.pos_emb.shape[1]}.')

        h = self.frame_in(motion) + self.pos_emb[:, :T, :]
        if cond is not None:
            h = h + self.cond_mlp(cond).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~valid_mask)
        probs = torch.sigmoid(self.out(h).squeeze(-1)) * valid_mask.float()
        return _budget_topk_straight_through(probs, valid_mask, self.budget_ratio or 0.1)

    def compute_auxiliary_loss(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor],
        probs: torch.Tensor,
        st_mask: torch.Tensor,
        oracle_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del motion, cond, st_mask
        if oracle_target is None:
            return torch.zeros((), device=probs.device)
        target = oracle_target * valid_mask.float()
        return ((probs - target) ** 2 * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)


class RetrievalGainKeyframeSelector(InformationGainKeyframeSelector):
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
        threshold: float = 0.5,
        budget_ratio: float = 0.1,
        **_: object,
    ):
        BaseKeyframeSelector.__init__(self, threshold=threshold, budget_ratio=budget_ratio)
        self.frame_in = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.frame_proj = nn.Linear(d_model, d_model)
        self.motion_proj = nn.Linear(d_model, d_model)
        self.score_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.score_head[-1].bias, -2.2)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, T, _ = motion.shape
        if T > self.pos_emb.shape[1]:
            raise ValueError(f'Sequence length {T} exceeds selector max_len {self.pos_emb.shape[1]}.')

        h = self.frame_in(motion) + self.pos_emb[:, :T, :]
        cond_embed = None
        if cond is not None:
            cond_embed = self.cond_mlp(cond)
            h = h + cond_embed.unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~valid_mask)

        frame_repr = F.normalize(self.frame_proj(h), dim=-1)
        motion_summary = _masked_mean(h, valid_mask)
        motion_repr = F.normalize(self.motion_proj(motion_summary), dim=-1).unsqueeze(1)

        if cond_embed is not None:
            cond_repr = F.normalize(cond_embed, dim=-1).unsqueeze(1)
            text_gain = (frame_repr * cond_repr).sum(dim=-1)
        else:
            cond_repr = torch.zeros_like(motion_repr)
            text_gain = torch.zeros(h.shape[:2], device=h.device, dtype=h.dtype)

        # Retrieval-gain should favor frames that both align with the text and
        # add information beyond the sequence-average motion descriptor.
        novelty_gain = (frame_repr - motion_repr).norm(dim=-1)
        score_input = torch.cat(
            [
                h,
                h * cond_repr.expand_as(h),
                h * motion_repr.expand_as(h),
            ],
            dim=-1,
        )
        base_logits = self.score_head(score_input).squeeze(-1)
        probs = torch.sigmoid(base_logits + 0.75 * text_gain + 0.35 * novelty_gain) * valid_mask.float()
        return _budget_topk_straight_through(probs, valid_mask, self.budget_ratio or 0.1)


class TextAlignmentKeyframeSelector(BaseKeyframeSelector):
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
        threshold: float = 0.5,
        **_: object,
    ):
        super().__init__(threshold=threshold)
        self.frame_in = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.score_head = nn.Linear(d_model, 1)
        self.motion_proj = nn.Linear(d_model, cond_dim)
        nn.init.constant_(self.score_head.bias, -2.2)
        self._last_alignment: Optional[torch.Tensor] = None

    def forward(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, T, _ = motion.shape
        if T > self.pos_emb.shape[1]:
            raise ValueError(f'Sequence length {T} exceeds selector max_len {self.pos_emb.shape[1]}.')

        h = self.frame_in(motion) + self.pos_emb[:, :T, :]
        cond_embed = None
        if cond is not None:
            cond_embed = self.cond_mlp(cond)
            h = h + cond_embed.unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~valid_mask)
        base_logits = self.score_head(h).squeeze(-1)

        if cond is not None:
            frame_proj = F.normalize(self.motion_proj(h), dim=-1)
            cond_norm = F.normalize(cond, dim=-1).unsqueeze(1)
            alignment = (frame_proj * cond_norm).sum(dim=-1)
        else:
            alignment = torch.zeros_like(base_logits)

        self._last_alignment = alignment.detach()
        probs = torch.sigmoid(base_logits + alignment) * valid_mask.float()
        return _straight_through(probs, self.threshold, valid_mask)

    def compute_auxiliary_loss(
        self,
        motion: torch.Tensor,
        valid_mask: torch.Tensor,
        cond: Optional[torch.Tensor],
        probs: torch.Tensor,
        st_mask: torch.Tensor,
        oracle_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del motion, cond, st_mask, oracle_target
        if self._last_alignment is None:
            return torch.zeros((), device=probs.device)
        target = _normalize_scores(torch.relu(self._last_alignment), valid_mask)
        return ((probs - target) ** 2 * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)


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
        local_max = F.max_pool1d(saliency.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
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