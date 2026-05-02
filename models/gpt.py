"""GPT models for autoregressive motion generation.

Conditioning design:
  Text embeddings are injected at *every* transformer block via a cross-attention
  sub-layer (self-attention → cross-attention → FFN), rather than being prepended
  as a single token.  This gives the model a persistent text signal throughout
  the sequence and avoids positional-encoding entanglement with the condition.
"""

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


class CrossAttention(nn.Module):
    """Multi-head cross-attention for injecting text conditioning context."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        S = context.shape[1]

        q = self.q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if context_mask is not None:
            attn = attn.masked_fill(~context_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class GPTBlock(nn.Module):
    """Transformer block: causal self-attention → cross-attention (text) → FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_ca = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.cross_attn(self.ln_ca(x), context, context_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MotionGPT(nn.Module):
    """Autoregressive GPT for motion-token generation conditioned on text.

    Text conditioning is supplied as a (B, cond_dim) CLIP embedding which is
    projected to (B, 1, d_model) and injected via cross-attention at every
    transformer layer.  This replaces the older single-prepended-token design,
    giving the model a persistent text signal at every depth.
    """

    def __init__(
        self,
        codebook_size: int,
        d_model: int = 512,
        n_layers: int = 12,
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

        # Project CLIP embedding → single cross-attention context token (B, 1, d_model)
        self.cond_tokens = 4

        # Project CLIP embedding → multiple context tokens (B, cond_tokens, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model * self.cond_tokens),
            nn.SiLU(),
            nn.Linear(d_model * self.cond_tokens, d_model * self.cond_tokens),
        )
        self.cond_seq_proj = nn.Linear(cond_dim, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight  # Weight tying

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) long tensor
            cond:   (B, cond_dim) text CLIP embedding
            mask:   (B, T) bool mask (True = valid position)
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tokens.shape

        tok_emb = self.token_emb(tokens)
        pos_emb = self.pos_emb[:, :T, :]
        h = tok_emb + pos_emb

        if cond.dim() == 2:
            context = self.cond_proj(cond).view(B, self.cond_tokens, self.d_model)
            if cond_mask is None:
                cond_mask = torch.ones(B, self.cond_tokens, dtype=torch.bool, device=tokens.device)
        elif cond.dim() == 3:
            context = self.cond_seq_proj(cond)
            if cond_mask is None:
                cond_mask = torch.ones(B, context.shape[1], dtype=torch.bool, device=tokens.device)
        else:
            raise ValueError(f'Expected cond to have rank 2 or 3, got shape {tuple(cond.shape)}')

        for block in self.blocks:
            h = block(h, context, mask, cond_mask)

        h = self.ln_f(h)
        return self.head(h)  # (B, T, vocab_size)

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
        cond_mask: Optional[torch.Tensor] = None,
        cond_uncond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = cond.shape[0]
        device = cond.device

        tokens = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self(tokens, cond, cond_mask=cond_mask)[:, -1, :]

            if guidance_scale > 0 and cond_uncond is not None:
                logits_uncond = self(tokens, cond_uncond, cond_mask=cond_uncond_mask)[:, -1, :]
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
        # Use 0 (a valid codebook index) as padding so decode_indices never
        # receives an out-of-range index that could trigger a CUDA assert.
        padded = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(result):
            if len(seq) > 0:
                padded[i, :len(seq)] = seq

        return padded
