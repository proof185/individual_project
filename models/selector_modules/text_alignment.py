from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseKeyframeSelector, _normalize_scores, _straight_through


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
        if cond is not None:
            h = h + self.cond_mlp(cond).unsqueeze(1)
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