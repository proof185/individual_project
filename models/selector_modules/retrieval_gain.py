from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseKeyframeSelector, _budget_topk_straight_through, _masked_mean
from .information_gain import InformationGainKeyframeSelector


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