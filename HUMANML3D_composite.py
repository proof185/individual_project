# %% [markdown]
# # Composite Model: Autoregressive + Diffusion In-Betweening
# 
# This notebook implements a **two-stage composite model** for text-to-motion generation:
# 
# ## Architecture Overview
# 
# **Stage 1: Autoregressive Keyframe Generation (MotionGPT-style)**
# - VQ-VAE tokenizes motion into discrete codes
# - GPT autoregressively generates motion tokens conditioned on text
# - Generates sparse keyframes every `n=5` frames
# 
# **Stage 2: Diffusion In-Betweening (MDM-style)**
# - Takes keyframes from Stage 1 as conditioning
# - Uses diffusion to fill in intermediate frames
# - Produces smooth, temporally coherent motion
# 
# ## Motivation
# - **AR models** excel at generating coherent global structure and long-range dependencies
# - **Diffusion models** excel at local refinement and smooth interpolation
# - Combining both leverages the strengths of each approach
# 
# ## Pipeline
# ```
# Text → [CLIP] → [MotionGPT] → Sparse Keyframes (every n frames)
#                                       ↓
#                      [Diffusion In-Betweening Model]
#                                       ↓
#                             Full Dense Motion Sequence
# ```

# %%
import os
import math
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# %%
# -----------------------------
# Config
# -----------------------------

@dataclass
class CompositeConfig:
    root: str = 'humanml'
    split: str = 'train'
    max_len: int = 196
    batch_size: int = 32
    num_workers: int = 0

    lr: float = 1e-4
    vqvae_steps: int = 100_000   # Stage 1a: VQ-VAE training
    gpt_steps: int = 200_000     # Stage 1b: GPT training
    inbetween_steps: int = 200_000  # Stage 2: Diffusion in-betweening
    grad_clip: float = 1.0

    # In-betweening config
    keyframe_interval: int = 5   # Generate keyframe every n frames

    # VQ-VAE
    codebook_size: int = 512
    codebook_dim: int = 512
    commitment_cost: float = 0.25
    downsample_rate: int = 4

    # Diffusion
    T_diffusion: int = 1000

    # Classifier-free guidance
    p_uncond: float = 0.1
    guidance_scale: float = 2.5

    # Model dimensions
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1

cfg = CompositeConfig()
print(f"Keyframe interval: every {cfg.keyframe_interval} frames")

# %%
# -----------------------------
# Load dataset normalization stats
# -----------------------------

mean_path = os.path.join(cfg.root, 'Mean.npy')
std_path  = os.path.join(cfg.root, 'Std.npy')

mean = torch.from_numpy(np.load(mean_path)).float()
std  = torch.from_numpy(np.load(std_path)).float()

mean = mean.view(-1)
std  = std.view(-1)

Fdim = mean.shape[0]
print('Feature dim:', Fdim)

# %%
# -----------------------------
# Dataset: HumanML3D with keyframe extraction for in-betweening
# -----------------------------

class HUMANML3DCompositeDataset(Dataset):
    """
    Dataset that provides both full motion and keyframe-extracted versions
    for training the composite model.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        max_len: Optional[int] = None,
        normalize: bool = True,
        use_cache: bool = True,
        text_encoder: Optional[callable] = None,
        keyframe_interval: int = 5,
    ):
        self.root = root
        self.split = split
        self.max_len = max_len
        self.normalize = normalize
        self.text_encoder = text_encoder
        self.keyframe_interval = keyframe_interval

        self.motion_dir = os.path.join(root, 'new_joint_vecs')
        self.text_dir = os.path.join(root, 'texts')
        self.split_file = os.path.join(root, f'{split}.txt')

        if not os.path.exists(self.motion_dir):
            raise FileNotFoundError(f"Missing {self.motion_dir}")

        with open(self.split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

        motion_cache = f"HUMANML3D_joint_vecs_{split}_maxlen{max_len}_norm{normalize}.pt"
        self.motion_cache_path = os.path.join(root, motion_cache)
        self.emb_cache_path = os.path.join(root, f'clip_embeddings_{split}.pt')

        self.data: List[Dict] = []
        self.embeddings: Dict[str, torch.Tensor] = {}

        if use_cache and os.path.exists(self.motion_cache_path):
            self.data = torch.load(self.motion_cache_path)
        else:
            self._build_motion_cache()
            if use_cache:
                torch.save(self.data, self.motion_cache_path)

        if use_cache and os.path.exists(self.emb_cache_path) and text_encoder is not None:
            print(f'Loading cached CLIP embeddings from {self.emb_cache_path}')
            cpu_cache = torch.load(self.emb_cache_path)
            for mid, emb in cpu_cache.items():
                self.embeddings[mid] = emb.to(device)
            print(f'Loaded {len(self.embeddings)} embeddings')
        elif text_encoder is not None:
            self._build_embedding_cache()
            if use_cache:
                cpu_cache = {k: v.cpu() for k, v in self.embeddings.items()}
                torch.save(cpu_cache, self.emb_cache_path)

    def _build_motion_cache(self):
        print('Building motion cache...')
        self.data = []
        for mid in self.ids:
            mpath = os.path.join(self.motion_dir, f'{mid}.npy')
            if not os.path.exists(mpath):
                continue
            motion = np.load(mpath).astype(np.float32)

            if motion.ndim != 2:
                continue

            if self.max_len is not None:
                motion = motion[:self.max_len]

            tpath = os.path.join(self.text_dir, f'{mid}.txt')
            texts = ['']
            if os.path.exists(tpath):
                with open(tpath, 'r', encoding='utf-8') as tf:
                    texts = [ln.strip() for ln in tf if ln.strip()]
                if len(texts) == 0:
                    texts = ['']

            self.data.append({
                'id': mid,
                'motion': torch.from_numpy(motion),
                'texts': texts,
                'length': motion.shape[0],
            })

    @torch.no_grad()
    def _build_embedding_cache(self):
        print('Pre-computing CLIP embeddings...')
        for item in self.data:
            mid = item['id']
            texts = item['texts']
            embs = self.text_encoder(texts, normalize=True)
            self.embeddings[mid] = embs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item['motion'].clone()

        if self.normalize:
            motion = (motion - mean) / (std + 1e-8)

        text_idx = random.randint(0, len(item['texts']) - 1)

        # Extract keyframes (every n frames)
        T = motion.shape[0]
        keyframe_indices = list(range(0, T, self.keyframe_interval))
        # Always include the last frame
        if keyframe_indices[-1] != T - 1:
            keyframe_indices.append(T - 1)
        keyframe_indices = torch.tensor(keyframe_indices, dtype=torch.long)
        keyframes = motion[keyframe_indices]  # (K, F)

        return {
            'id': item['id'],
            'motion': motion,
            'length': item['length'],
            'text_idx': text_idx,
            'keyframes': keyframes,
            'keyframe_indices': keyframe_indices,
        }

    def get_embedding(self, sample_id: str, text_idx: int) -> torch.Tensor:
        return self.embeddings[sample_id][text_idx]


def collate_composite(batch: List[Dict]):
    motions = [b['motion'] for b in batch]
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    ids = [b['id'] for b in batch]
    text_idxs = [b['text_idx'] for b in batch]
    keyframes_list = [b['keyframes'] for b in batch]
    keyframe_indices_list = [b['keyframe_indices'] for b in batch]

    B = len(motions)
    T_max = int(lengths.max())
    F = motions[0].shape[1]

    x = torch.zeros(B, T_max, F, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, m in enumerate(motions):
        T = m.shape[0]
        x[i, :T] = m
        mask[i, :T] = True

    # Pad keyframes
    K_max = max(kf.shape[0] for kf in keyframes_list)
    keyframes = torch.zeros(B, K_max, F, dtype=torch.float32)
    keyframe_mask = torch.zeros(B, K_max, dtype=torch.bool)
    keyframe_indices = torch.zeros(B, K_max, dtype=torch.long)

    for i, (kf, ki) in enumerate(zip(keyframes_list, keyframe_indices_list)):
        K = kf.shape[0]
        keyframes[i, :K] = kf
        keyframe_mask[i, :K] = True
        keyframe_indices[i, :K] = ki

    return {
        'motion': x,
        'mask': mask,
        'lengths': lengths,
        'ids': ids,
        'text_idxs': text_idxs,
        'keyframes': keyframes,
        'keyframe_mask': keyframe_mask,
        'keyframe_indices': keyframe_indices,
    }

# %%
# -----------------------------
# CLIP text encoder (frozen)
# -----------------------------

clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

@torch.no_grad()
def encode_text(texts: List[str], normalize: bool = True) -> torch.Tensor:
    tokens = clip.tokenize(texts, truncate=True).to(device)
    emb = clip_model.encode_text(tokens).float()
    if normalize:
        emb = F.normalize(emb, dim=-1)
    return emb

empty_emb = encode_text(['']).squeeze(0)
print('Empty embedding norm:', empty_emb.norm().item())

# %%
# -----------------------------
# Create dataset
# -----------------------------

dataset = HUMANML3DCompositeDataset(
    cfg.root,
    split='train',
    max_len=cfg.max_len,
    normalize=True,
    use_cache=True,
    text_encoder=encode_text,
    keyframe_interval=cfg.keyframe_interval,
)
print('Dataset size:', len(dataset))

loader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    collate_fn=collate_composite,
    drop_last=True
)

batch = next(iter(loader))
print('Motion:', batch['motion'].shape)
print('Keyframes:', batch['keyframes'].shape)
print('Keyframe indices sample:', batch['keyframe_indices'][0, :10])

# %% [markdown]
# ## Part 1: VQ-VAE for Motion Tokenization
# 
# The VQ-VAE discretizes continuous motion into tokens that the GPT can model.

# %%
# -----------------------------
# VQ-VAE: Vector Quantization Variational Autoencoder
# -----------------------------

class VectorQuantizer(nn.Module):
    """Vector Quantization with EMA codebook updates."""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)

        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )

        indices = d.argmin(dim=1)
        z_q = self.embedding(indices).view(B, T, D)

        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_embeddings).float()
                self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
                dw = z_flat.t() @ encodings
                self.ema_w.mul_(self.decay).add_(dw.t(), alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        commitment_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()  # Straight-through

        indices = indices.view(B, T)
        return z_q, indices, self.commitment_cost * commitment_loss


class MotionVQVAE(nn.Module):
    """VQ-VAE for motion tokenization."""
    def __init__(
        self,
        feature_dim: int,
        codebook_size: int = 512,
        codebook_dim: int = 512,
        downsample_rate: int = 4,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_dim = codebook_dim
        self.downsample_rate = downsample_rate

        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, codebook_dim, kernel_size=3, padding=1),
        )

        self.vq = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Conv1d(codebook_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, feature_dim, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        z = self.encoder(x)
        z = z.transpose(1, 2)
        z_q, indices, _ = self.vq(z)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z_q = z_q.transpose(1, 2)
        x = self.decoder(z_q)
        x = x.transpose(1, 2)
        return x

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.vq.embedding(indices)
        return self.decode(z_q)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_conv = x.transpose(1, 2)
        z = self.encoder(x_conv)
        z = z.transpose(1, 2)

        z_q, indices, vq_loss = self.vq(z)
        x_recon = self.decode(z_q)

        return x_recon, indices, vq_loss


vqvae = MotionVQVAE(
    feature_dim=Fdim,
    codebook_size=cfg.codebook_size,
    codebook_dim=cfg.codebook_dim,
    downsample_rate=cfg.downsample_rate,
    commitment_cost=cfg.commitment_cost,
).to(device)

print(f'VQ-VAE params: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M')

# %% [markdown]
# ## Part 2: MotionGPT for Autoregressive Generation
# 
# The GPT model generates motion tokens autoregressively, conditioned on text.

# %%
# -----------------------------
# GPT Model: Autoregressive Motion Token Generation
# -----------------------------

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
    """GPT for autoregressive motion token generation."""
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


gpt = MotionGPT(
    codebook_size=cfg.codebook_size,
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    dropout=cfg.dropout,
    max_len=cfg.max_len // cfg.downsample_rate + 10,
    cond_dim=512,
).to(device)

print(f'MotionGPT params: {sum(p.numel() for p in gpt.parameters())/1e6:.2f}M')

# %% [markdown]
# ## Part 3: Diffusion In-Betweening Model
# 
# This is the key new component: a diffusion model that takes sparse keyframes and fills in the intermediate frames.
# 
# **Architecture:**
# - Input: Keyframe positions + noisy motion sequence
# - Conditioning: Keyframe values are injected at their positions
# - Output: Predicted clean motion (x0-prediction)
# 
# **Training:**
# - During training, we extract keyframes from GT motion
# - The model learns to reconstruct the full sequence from sparse keyframes
# 
# **Inference:**
# - Keyframes come from the AR model
# - Diffusion fills in the gaps

# %%
# -----------------------------
# Diffusion Utilities
# -----------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


class InbetweenDiffusion:
    """
    Diffusion process for in-betweening.
    Key difference: keyframes are kept fixed during sampling.
    """
    def __init__(self, T: int):
        self.T = T
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register(betas=betas, alphas=alphas, alpha_bar=alpha_bar)

    def register(self, **tensors):
        for k, v in tensors.items():
            setattr(self, k, v.to(device))

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        shape = [B] + [1] * (x0.ndim - 1)
        s1 = self.sqrt_alpha_bar[t].view(*shape)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(*shape)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample_inbetween(
        self,
        model,
        xt: torch.Tensor,
        t: int,
        cond: torch.Tensor,
        mask: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
        guidance_scale: float = 0.0,
        cond_uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single denoising step with keyframe conditioning.
        Keyframes are kept fixed (replaced after each step).
        """
        B = xt.shape[0]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict x0
        if guidance_scale > 0 and cond_uncond is not None:
            x0_uncond = model(xt, t_batch, cond_uncond, mask, keyframes, keyframe_indices, keyframe_mask)
            x0_cond = model(xt, t_batch, cond, mask, keyframes, keyframe_indices, keyframe_mask)
            x0_hat = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        else:
            x0_hat = model(xt, t_batch, cond, mask, keyframes, keyframe_indices, keyframe_mask)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        if t == 0:
            x_prev = x0_hat
        else:
            coef1 = torch.sqrt(self.alpha_bar[t - 1]) * beta_t / (1 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1 - self.alpha_bar[t - 1]) / (1 - alpha_bar_t)
            mean = coef1 * x0_hat + coef2 * xt
            var = beta_t * (1 - self.alpha_bar[t - 1]) / (1 - alpha_bar_t)
            noise = torch.randn_like(xt)
            x_prev = mean + torch.sqrt(var) * noise

        # Replace keyframe positions with actual keyframe values
        x_prev = self._replace_keyframes(x_prev, keyframes, keyframe_indices, keyframe_mask)

        x_prev = x_prev * mask.float().unsqueeze(-1)
        return x_prev

    def _replace_keyframes(
        self,
        x: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace positions in x with keyframe values."""
        B, T, F = x.shape
        x_out = x.clone()

        for b in range(B):
            valid_mask = keyframe_mask[b]
            valid_indices = keyframe_indices[b][valid_mask]
            valid_keyframes = keyframes[b][valid_mask]
            x_out[b, valid_indices] = valid_keyframes

        return x_out

    @torch.no_grad()
    def sample_inbetween(
        self,
        model,
        shape: Tuple[int, int, int],
        cond: torch.Tensor,
        mask: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
        guidance_scale: float = 2.5,
        cond_uncond: Optional[torch.Tensor] = None,
    ):
        """
        Full sampling loop for in-betweening.
        """
        x = torch.randn(shape, device=device)

        # Initialize with keyframes
        x = self._replace_keyframes(x, keyframes, keyframe_indices, keyframe_mask)
        x = x * mask.float().unsqueeze(-1)

        for t in reversed(range(self.T)):
            x = self.p_sample_inbetween(
                model, x, t, cond, mask,
                keyframes, keyframe_indices, keyframe_mask,
                guidance_scale=guidance_scale,
                cond_uncond=cond_uncond
            )

        return x


diff_inbetween = InbetweenDiffusion(cfg.T_diffusion)

# %%
# -----------------------------
# In-Betweening Transformer Model
# -----------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class InbetweenTransformer(nn.Module):
    """
    Transformer for diffusion in-betweening.
    
    Key features:
    - Accepts keyframe conditioning via special tokens
    - Learns to interpolate between keyframes
    - Predicts x0 (clean motion)
    """
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int = 512,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        # Input projection
        self.frame_in = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Keyframe indicator embedding (learnable)
        # 0 = not keyframe, 1 = keyframe
        self.keyframe_emb = nn.Embedding(2, d_model)

        # Keyframe value encoder (separate from noisy input)
        self.keyframe_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Text condition
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer
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

        # Output
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        keyframes: torch.Tensor,
        keyframe_indices: torch.Tensor,
        keyframe_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xt: Noisy motion (B, T, F)
            t: Diffusion timestep (B,)
            cond: Text condition (B, cond_dim)
            mask: Valid frame mask (B, T)
            keyframes: Keyframe values (B, K, F)
            keyframe_indices: Positions of keyframes (B, K)
            keyframe_mask: Valid keyframe mask (B, K)
        Returns:
            x0_hat: Predicted clean motion (B, T, F)
        """
        B, T, F = xt.shape

        # Create keyframe indicator for each frame
        keyframe_indicator = torch.zeros(B, T, dtype=torch.long, device=xt.device)
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            keyframe_indicator[b, valid_idx] = 1

        # Encode noisy input
        h = self.frame_in(xt)  # (B, T, d)
        h = h + self.pos_emb[:, :T, :]

        # Add keyframe indicator embedding
        h = h + self.keyframe_emb(keyframe_indicator)

        # Inject keyframe values at their positions
        keyframe_features = self.keyframe_encoder(keyframes)  # (B, K, d)
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            h[b, valid_idx] = h[b, valid_idx] + keyframe_features[b, valid]

        # Time + condition token
        t_emb = timestep_embedding(t, self.d_model)
        zt = self.t_mlp(t_emb) + self.c_mlp(cond)
        zt = zt.unsqueeze(1)  # (B, 1, d)

        # Prepend conditioning token
        tokens = torch.cat([zt, h], dim=1)  # (B, 1+T, d)

        # Padding mask
        src_key_padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
            ~mask
        ], dim=1)

        # Transformer
        y = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        y = y[:, 1:, :]  # Remove condition token

        # Output prediction
        x0_hat = self.out(y)
        x0_hat = x0_hat * mask.float().unsqueeze(-1)

        return x0_hat


inbetween_model = InbetweenTransformer(
    feature_dim=Fdim,
    cond_dim=512,
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    dropout=cfg.dropout,
    max_len=cfg.max_len + 10,
).to(device)

print(f'In-betweening model params: {sum(p.numel() for p in inbetween_model.parameters())/1e6:.2f}M')

# %% [markdown]
# ## Training Stage 1a: VQ-VAE

# %%
# -----------------------------
# Training utilities
# -----------------------------

def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    se = (a - b) ** 2
    se = se * m
    denom = m.sum() * se.shape[-1] + 1e-8
    return se.sum() / denom


def vel_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[:, 1:] & mask[:, :-1]
    v_gt = x[:, 1:] - x[:, :-1]
    v_pr = x_hat[:, 1:] - x_hat[:, :-1]
    return masked_mse(v_gt, v_pr, m)

# %%
# -----------------------------
# Stage 1a: VQ-VAE Training
# -----------------------------

from itertools import cycle

vqvae_optimizer = torch.optim.AdamW(vqvae.parameters(), lr=cfg.lr)

lambda_recon = 1.0
lambda_vel = 0.5

# Check if final checkpoint exists
vqvae_final_ckpt_path = f'checkpoints/composite_vqvae_step{cfg.vqvae_steps}.pt'
if os.path.exists(vqvae_final_ckpt_path):
    print(f"Loading VQ-VAE from final checkpoint: {vqvae_final_ckpt_path}")
    vqvae_ckpt = torch.load(vqvae_final_ckpt_path, map_location=device)
    vqvae.load_state_dict(vqvae_ckpt['model'])
    vqvae_optimizer.load_state_dict(vqvae_ckpt['optimizer'])
    print("VQ-VAE training already complete! Loaded from checkpoint.")
else:
    vqvae.train()
    data_iter = cycle(loader)

    print("Stage 1a: Training VQ-VAE...")

    for step in range(1, cfg.vqvae_steps + 1):
        batch = next(data_iter)
        x = batch['motion'].to(device)
        mask = batch['mask'].to(device)

        B, T, _ = x.shape
        pad_len = (cfg.downsample_rate - T % cfg.downsample_rate) % cfg.downsample_rate
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len), value=False)

        x_recon, indices, vq_loss = vqvae(x, mask)

        x_recon = x_recon[:, :T]
        mask_orig = batch['mask'].to(device)

        recon_loss = masked_mse(batch['motion'].to(device), x_recon, mask_orig)
        v_loss = vel_loss(batch['motion'].to(device), x_recon, mask_orig)

        loss = lambda_recon * recon_loss + vq_loss + lambda_vel * v_loss

        vqvae_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(vqvae.parameters(), cfg.grad_clip)
        vqvae_optimizer.step()

        if step % 200 == 0:
            print(f"step {step:>7d} | recon {recon_loss.item():.5f} | vq {vq_loss.item():.5f} | vel {v_loss.item():.5f}")

        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model': vqvae.state_dict(),
                'optimizer': vqvae_optimizer.state_dict(),
                'step': step,
            }, f'checkpoints/composite_vqvae_step{step}.pt')
            print('Saved VQ-VAE checkpoint')

    print("VQ-VAE training complete!")

# %% [markdown]
# ## Training Stage 1b: MotionGPT

# %%
# -----------------------------
# Stage 1b: GPT Training
# -----------------------------

def prepare_gpt_batch(batch, vqvae, downsample_rate):
    x = batch['motion'].to(device)
    mask = batch['mask'].to(device)
    B, T, _ = x.shape

    pad_len = (downsample_rate - T % downsample_rate) % downsample_rate
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))

    with torch.no_grad():
        indices, _ = vqvae.encode(x)

    T_tokens = indices.shape[1]
    token_lengths = (batch['lengths'].to(device) + downsample_rate - 1) // downsample_rate
    token_mask = torch.arange(T_tokens, device=device).unsqueeze(0) < token_lengths.unsqueeze(1)

    bos = torch.full((B, 1), gpt.bos_token, dtype=torch.long, device=device)
    eos = torch.full((B, 1), gpt.eos_token, dtype=torch.long, device=device)

    input_tokens = torch.cat([bos, indices], dim=1)
    target_tokens = torch.cat([indices, eos], dim=1)
    target_mask = torch.cat([token_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)

    return input_tokens, target_tokens, target_mask

# %%
# Load VQ-VAE checkpoint if needed
# vqvae_ckpt = torch.load('checkpoints/composite_vqvae_step100000.pt', map_location=device)
# vqvae.load_state_dict(vqvae_ckpt['model'])

vqvae.eval()

gpt_optimizer = torch.optim.AdamW(gpt.parameters(), lr=cfg.lr)

# Check if final checkpoint exists
gpt_final_ckpt_path = f'checkpoints/composite_gpt_step{cfg.gpt_steps}.pt'
if os.path.exists(gpt_final_ckpt_path):
    print(f"Loading GPT from final checkpoint: {gpt_final_ckpt_path}")
    gpt_ckpt = torch.load(gpt_final_ckpt_path, map_location=device)
    gpt.load_state_dict(gpt_ckpt['gpt'])
    vqvae.load_state_dict(gpt_ckpt['vqvae'])
    gpt_optimizer.load_state_dict(gpt_ckpt['optimizer'])
    print("MotionGPT training already complete! Loaded from checkpoint.")
else:
    gpt.train()
    data_iter = cycle(loader)

    print("Stage 1b: Training MotionGPT...")

    for step in range(1, cfg.gpt_steps + 1):
        batch = next(data_iter)

        input_tokens, target_tokens, target_mask = prepare_gpt_batch(batch, vqvae, cfg.downsample_rate)

        cond_list = []
        for mid, tidx in zip(batch['ids'], batch['text_idxs']):
            if random.random() < cfg.p_uncond:
                cond_list.append(empty_emb)
            else:
                cond_list.append(dataset.get_embedding(mid, tidx))
        cond = torch.stack(cond_list)

        logits = gpt(input_tokens, cond)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            target_tokens.reshape(-1),
            ignore_index=gpt.pad_token,
            reduction='none'
        ).reshape(B, T)

        loss = (loss * target_mask.float()).sum() / (target_mask.float().sum() + 1e-8)

        gpt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(gpt.parameters(), cfg.grad_clip)
        gpt_optimizer.step()

        if step % 200 == 0:
            ppl = torch.exp(loss).item()
            print(f"step {step:>7d} | loss {loss.item():.5f} | ppl {ppl:.2f}")

        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'gpt': gpt.state_dict(),
                'vqvae': vqvae.state_dict(),
                'optimizer': gpt_optimizer.state_dict(),
                'step': step,
            }, f'checkpoints/composite_gpt_step{step}.pt')
            print('Saved GPT checkpoint')

    print("MotionGPT training complete!")

# %% [markdown]
# ## Training Stage 2: Diffusion In-Betweening

# %%
# -----------------------------
# Stage 2: Diffusion In-Betweening Training
# -----------------------------

inbetween_optimizer = torch.optim.AdamW(inbetween_model.parameters(), lr=cfg.lr)

# Check if final checkpoint exists
inbetween_final_ckpt_path = f'checkpoints/composite_inbetween_step{cfg.inbetween_steps}.pt'
if os.path.exists(inbetween_final_ckpt_path):
    print(f"Loading In-Betweening model from final checkpoint: {inbetween_final_ckpt_path}")
    inbetween_ckpt = torch.load(inbetween_final_ckpt_path, map_location=device)
    inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
    inbetween_optimizer.load_state_dict(inbetween_ckpt['optimizer'])
    print("Diffusion in-betweening training already complete! Loaded from checkpoint.")
else:
    inbetween_model.train()
    data_iter = cycle(loader)

    print(f"Stage 2: Training Diffusion In-Betweening (keyframe interval={cfg.keyframe_interval})...")

    for step in range(1, cfg.inbetween_steps + 1):
        batch = next(data_iter)
        x0 = batch['motion'].to(device)
        mask = batch['mask'].to(device)
        keyframes = batch['keyframes'].to(device)
        keyframe_indices = batch['keyframe_indices'].to(device)
        keyframe_mask = batch['keyframe_mask'].to(device)

        # Get text condition (with CFG dropout)
        cond_list = []
        for mid, tidx in zip(batch['ids'], batch['text_idxs']):
            if random.random() < cfg.p_uncond:
                cond_list.append(empty_emb)
            else:
                cond_list.append(dataset.get_embedding(mid, tidx))
        cond = torch.stack(cond_list)

        # Sample diffusion timestep
        B = x0.shape[0]
        t = torch.randint(0, cfg.T_diffusion, (B,), device=device)

        # Add noise
        noise = torch.randn_like(x0)
        xt = diff_inbetween.q_sample(x0, t, noise)
        xt = xt * mask.float().unsqueeze(-1)

        # Replace keyframe positions with clean values (they're given)
        xt = diff_inbetween._replace_keyframes(xt, keyframes, keyframe_indices, keyframe_mask)

        # Predict x0
        x0_hat = inbetween_model(xt, t, cond, mask, keyframes, keyframe_indices, keyframe_mask)

        # Loss on non-keyframe positions (keyframes are given, no need to predict)
        # Create mask for non-keyframe positions
        non_keyframe_mask = mask.clone()
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            non_keyframe_mask[b, valid_idx] = False

        loss = masked_mse(x0, x0_hat, non_keyframe_mask)

        # Also add small loss on keyframes to ensure consistency
        keyframe_loss = masked_mse(x0, x0_hat, mask) * 0.1
        loss = loss + keyframe_loss

        # Optional: velocity smoothness
        v_loss = vel_loss(x0, x0_hat, mask) * 0.5
        loss = loss + v_loss

        inbetween_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(inbetween_model.parameters(), cfg.grad_clip)
        inbetween_optimizer.step()

        if step % 200 == 0:
            print(f"step {step:>7d} | loss {loss.item():.5f}")

        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'inbetween': inbetween_model.state_dict(),
                'optimizer': inbetween_optimizer.state_dict(),
                'step': step,
                'cfg': cfg.__dict__,
            }, f'checkpoints/composite_inbetween_step{step}.pt')
            print('Saved in-betweening checkpoint')

    print("Diffusion in-betweening training complete!")

# %% [markdown]
# ## Inference: Complete Pipeline
# 
# 1. Text → CLIP embedding
# 2. CLIP embedding → MotionGPT → Sparse keyframes (every n frames)
# 3. Keyframes → Diffusion In-Betweening → Full dense motion

# %%
# -----------------------------
# Load checkpoints for inference
# -----------------------------

# Uncomment to load from checkpoints:
# gpt_ckpt = torch.load('checkpoints/composite_gpt_step200000.pt', map_location=device)
# vqvae.load_state_dict(gpt_ckpt['vqvae'])
# gpt.load_state_dict(gpt_ckpt['gpt'])

# inbetween_ckpt = torch.load('checkpoints/composite_inbetween_step200000.pt', map_location=device)
# inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])

vqvae.eval()
gpt.eval()
inbetween_model.eval()

print("Models loaded for inference")

# %%
# -----------------------------
# Composite Generation Pipeline
# -----------------------------

@torch.no_grad()
def generate_composite(
    prompt: str,
    length: int = 100,
    keyframe_interval: int = 5,
    # AR generation params
    ar_temperature: float = 0.9,
    ar_top_k: int = 50,
    ar_top_p: float = 0.95,
    ar_guidance_scale: float = 2.5,
    # Diffusion params
    diff_guidance_scale: float = 2.5,
):
    """
    Complete composite generation pipeline.
    
    Args:
        prompt: Text description
        length: Target motion length in frames
        keyframe_interval: Generate keyframe every n frames
        ar_*: Autoregressive generation parameters
        diff_*: Diffusion in-betweening parameters
    
    Returns:
        motion: Generated motion (T, F) in original feature space
        keyframes: The sparse keyframes from AR model (K, F)
        keyframe_indices: Indices of keyframes
    """
    print(f"Generating motion for: '{prompt}'")
    print(f"Target length: {length} frames, keyframe interval: {keyframe_interval}")
    
    # ========== Stage 1: AR Keyframe Generation ==========
    print("\nStage 1: Generating keyframes with MotionGPT...")
    
    # Encode text
    cond = encode_text([prompt])
    cond_uncond = encode_text([''])
    
    # Calculate how many tokens we need from VQ-VAE
    # After VQ-VAE decoding, we get frames at downsample_rate resolution
    target_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate
    
    # Generate motion tokens autoregressively
    tokens = gpt.generate(
        cond=cond,
        max_new_tokens=target_tokens,
        temperature=ar_temperature,
        top_k=ar_top_k,
        top_p=ar_top_p,
        guidance_scale=ar_guidance_scale,
        cond_uncond=cond_uncond,
    )  # (1, T')
    
    # Decode tokens to continuous motion
    ar_motion_norm = vqvae.decode_indices(tokens)  # (1, T_decoded, F)
    ar_motion_norm = ar_motion_norm[0, :length]    # (T, F)
    
    print(f"AR output shape: {ar_motion_norm.shape}")
    
    # Extract sparse keyframes from AR output
    keyframe_indices = list(range(0, length, keyframe_interval))
    if keyframe_indices[-1] != length - 1:
        keyframe_indices.append(length - 1)
    keyframe_indices = torch.tensor(keyframe_indices, dtype=torch.long, device=device)
    
    keyframes = ar_motion_norm[keyframe_indices]  # (K, F)
    print(f"Extracted {len(keyframe_indices)} keyframes at positions: {keyframe_indices.tolist()[:10]}...")
    
    # ========== Stage 2: Diffusion In-Betweening ==========
    print("\nStage 2: Filling in-between frames with diffusion...")
    
    # Prepare for diffusion
    B, T, F = 1, length, Fdim
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    
    keyframes_batch = keyframes.unsqueeze(0)  # (1, K, F)
    keyframe_indices_batch = keyframe_indices.unsqueeze(0)  # (1, K)
    keyframe_mask_batch = torch.ones(1, len(keyframe_indices), dtype=torch.bool, device=device)
    
    # Run diffusion in-betweening
    motion_norm = diff_inbetween.sample_inbetween(
        model=inbetween_model,
        shape=(B, T, F),
        cond=cond,
        mask=mask,
        keyframes=keyframes_batch,
        keyframe_indices=keyframe_indices_batch,
        keyframe_mask=keyframe_mask_batch,
        guidance_scale=diff_guidance_scale,
        cond_uncond=cond_uncond,
    )  # (1, T, F)
    
    motion_norm = motion_norm[0]  # (T, F)
    
    # Unnormalize
    motion = motion_norm.cpu() * (std + 1e-8) + mean
    keyframes_unnorm = keyframes.cpu() * (std + 1e-8) + mean
    
    print(f"\nGeneration complete! Output shape: {motion.shape}")
    
    return motion, keyframes_unnorm, keyframe_indices.cpu()


# Also provide AR-only and diffusion-only baselines for comparison
@torch.no_grad()
def generate_ar_only(prompt: str, length: int = 100, guidance_scale: float = 2.5):
    """Generate using only the AR model (no diffusion refinement)."""
    cond = encode_text([prompt])
    cond_uncond = encode_text([''])
    
    target_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate
    
    tokens = gpt.generate(
        cond=cond,
        max_new_tokens=target_tokens,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        guidance_scale=guidance_scale,
        cond_uncond=cond_uncond,
    )
    
    motion_norm = vqvae.decode_indices(tokens)
    motion_norm = motion_norm[0, :length]
    motion = motion_norm.cpu() * (std + 1e-8) + mean
    
    return motion

# %%
# -----------------------------
# Generate sample motion
# -----------------------------

prompt = "a person walks forward and then jumps"

# Composite generation
motion, keyframes, keyframe_idx = generate_composite(
    prompt,
    length=120,
    keyframe_interval=cfg.keyframe_interval,
)

print(f"\nFinal motion shape: {motion.shape}")
print(f"Keyframes shape: {keyframes.shape}")
print(f"Keyframe indices: {keyframe_idx.tolist()}")

# Save
np.save("sample_composite_output.npy", motion.numpy())
np.save("sample_composite_keyframes.npy", keyframes.numpy())

# %%
# -----------------------------
# Visualization
# -----------------------------

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

HUMANML3D_EDGES = [
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (9, 13), (13, 16), (16, 18), (18, 20),
]

def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    T = data.shape[0]
    
    r_rot_vel = data[:, 0]
    r_rot_ang = np.zeros(T, dtype=np.float32)
    r_rot_ang[1:] = np.cumsum(r_rot_vel[:-1])
    
    r_pos = np.zeros((T, 3), dtype=np.float32)
    r_pos[:, 1] = data[:, 3]
    
    r_vel_local = np.zeros((T, 3), dtype=np.float32)
    r_vel_local[1:, 0] = data[:-1, 1]
    r_vel_local[1:, 2] = data[:-1, 2]
    
    cos_r = np.cos(r_rot_ang)
    sin_r = np.sin(r_rot_ang)
    r_vel_world = np.zeros_like(r_vel_local)
    r_vel_world[:, 0] = cos_r * r_vel_local[:, 0] - sin_r * r_vel_local[:, 2]
    r_vel_world[:, 2] = sin_r * r_vel_local[:, 0] + cos_r * r_vel_local[:, 2]
    
    r_pos[:, 0] = np.cumsum(r_vel_world[:, 0])
    r_pos[:, 2] = np.cumsum(r_vel_world[:, 2])
    
    ric = data[:, 4:4 + (joints_num - 1) * 3]
    ric = ric.reshape(T, joints_num - 1, 3)
    
    positions = np.zeros((T, joints_num - 1, 3), dtype=np.float32)
    positions[:, :, 0] = cos_r[:, None] * ric[:, :, 0] - sin_r[:, None] * ric[:, :, 2]
    positions[:, :, 1] = ric[:, :, 1]
    positions[:, :, 2] = sin_r[:, None] * ric[:, :, 0] + cos_r[:, None] * ric[:, :, 2]
    
    positions[:, :, 0] += r_pos[:, 0:1]
    positions[:, :, 2] += r_pos[:, 2:3]
    
    joints = np.concatenate([r_pos[:, None, :], positions], axis=1)
    return joints


def animate_skeleton(motion, edges=HUMANML3D_EDGES, title="Motion", stride=1, 
                     elev=15, azim=-70, interval=80, center=True,
                     keyframe_indices=None):
    motion = motion[::stride].copy()
    T, J, _ = motion.shape
    
    if keyframe_indices is not None:
        keyframe_indices = [i // stride for i in keyframe_indices if i // stride < T]
    
    if center:
        motion = motion - motion[:, [0], :]
    
    mins = motion.reshape(-1, 3).min(axis=0)
    maxs = motion.reshape(-1, 3).max(axis=0)
    span = (maxs - mins).max()
    center_pt = (maxs + mins) / 2
    half = span / 2 * 1.2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center_pt[0] - half, center_pt[0] + half)
    ax.set_ylim(center_pt[1] - half, center_pt[1] + half)
    ax.set_zlim(center_pt[2] - half, center_pt[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pts = ax.scatter([], [], [], s=20)
    lines = [ax.plot([], [], [], lw=2)[0] for _ in edges]

    def init():
        pts._offsets3d = ([], [], [])
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return [pts] + lines

    def update(t):
        frame = motion[t]
        xs, ys, zs = frame[:, 0], frame[:, 2], frame[:, 1]
        pts._offsets3d = (xs, ys, zs)
        
        # Color based on keyframe
        is_keyframe = keyframe_indices is not None and t in keyframe_indices
        color = 'red' if is_keyframe else 'blue'
        
        for k, (i, j) in enumerate(edges):
            lines[k].set_data([xs[i], xs[j]], [ys[i], ys[j]])
            lines[k].set_3d_properties([zs[i], zs[j]])
            lines[k].set_color(color)
        
        kf_str = " [KEYFRAME]" if is_keyframe else ""
        ax.set_title(f"{title} | frame {t+1}/{T}{kf_str}")
        return [pts] + lines

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval, blit=False)
    plt.close(fig)
    return anim

# %%
# Visualize the composite output
vecs = np.load("sample_composite_output.npy")
joints = recover_from_ric(vecs, joints_num=22)
print("Joints shape:", joints.shape)

# Show with keyframe highlighting
anim = animate_skeleton(
    joints, 
    edges=HUMANML3D_EDGES, 
    title=f"Composite: {prompt}", 
    center=True,
    keyframe_indices=keyframe_idx.tolist() if 'keyframe_idx' in dir() else None
)
display(HTML(anim.to_jshtml()))

# %% [markdown]
# ## Comparison: Composite vs AR-only
# 
# Generate both versions to compare quality.

# %%
# Compare AR-only vs Composite
comparison_prompt = "a person kicks with their right leg"
comparison_length = 100

print("Generating AR-only...")
ar_only_motion = generate_ar_only(comparison_prompt, length=comparison_length)

print("\nGenerating Composite (AR + Diffusion)...")
composite_motion, _, _ = generate_composite(comparison_prompt, length=comparison_length)

print(f"\nAR-only shape: {ar_only_motion.shape}")
print(f"Composite shape: {composite_motion.shape}")

# %%
# Visualize AR-only
ar_joints = recover_from_ric(ar_only_motion.numpy(), joints_num=22)
anim_ar = animate_skeleton(ar_joints, title=f"AR-only: {comparison_prompt}", center=True)
display(HTML(anim_ar.to_jshtml()))

# %%
# Visualize Composite
composite_joints = recover_from_ric(composite_motion.numpy(), joints_num=22)
anim_composite = animate_skeleton(composite_joints, title=f"Composite: {comparison_prompt}", center=True)
display(HTML(anim_composite.to_jshtml()))


