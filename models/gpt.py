"""GPT models for autoregressive motion generation."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MotionGPT(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 256,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.max_len = max_len

        self.bos_token = codebook_size
        self.eos_token = codebook_size + 1
        self.pad_token = codebook_size + 2
        self.vocab_size = codebook_size + 3

        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = tokens.shape

        tok_emb = self.token_emb(tokens)
        pos_emb = self.pos_emb[:, :T, :]
        h = tok_emb + pos_emb

        cond_token = self.cond_proj(cond).unsqueeze(1)
        h = torch.cat([cond_token, h], dim=1)

        if mask is not None:
            mask = torch.cat([
                torch.ones(B, 1, dtype=torch.bool, device=mask.device),
                mask
            ], dim=1)

        for block in self.blocks:
            h = block(h, mask)

        h = self.ln_f(h)
        logits = self.head(h)
        logits = logits[:, 1:, :]

        return logits

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        guidance_scale: float = 0.0,
        cond_uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = cond.shape[0]
        device = cond.device

        tokens = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self(tokens, cond)[:, -1, :]

            if guidance_scale > 0 and cond_uncond is not None:
                logits_uncond = self(tokens, cond_uncond)[:, -1, :]
                logits = logits_uncond + guidance_scale * (logits - logits_uncond)

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == self.eos_token).all():
                break

        result = []
        for i in range(B):
            seq = tokens[i, 1:]
            eos_pos = (seq == self.eos_token).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                seq = seq[:eos_pos[0]]
            seq = seq[seq < self.codebook_size]
            result.append(seq)

        max_len = max(len(s) for s in result) if result else 0
        padded = torch.full((B, max_len), self.pad_token, dtype=torch.long, device=device)
        for i, seq in enumerate(result):
            if len(seq) > 0:
                padded[i, :len(seq)] = seq

        return padded
