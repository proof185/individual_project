from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import BaseKeyframeSelector, _budget_topk_straight_through


class ReconstructionKeyframeSelector(BaseKeyframeSelector):
    """Trainable selector whose supervision comes from CondMDI reconstruction loss."""

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
        budget_ratio: float = 0.2,
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
        return _budget_topk_straight_through(probs, valid_mask, self.budget_ratio or 0.2)
