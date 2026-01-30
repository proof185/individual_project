# %% [markdown]
# # MotionGPT-style Autoregressive Language Model on HumanML3D `new_joint_vecs` (T, 263)
# 
# This notebook implements the core training setup inspired by **MotionGPT** (Jiang et al., 2023) on the **HumanML3D** dataset using the **redundant per-frame feature representation** (`new_joint_vecs`: `(frames, features)` where `features=263`).
# 
# **Architecture Overview (MotionGPT-style):**
# 1. **VQ-VAE (Vector Quantization Variational Autoencoder)**: Discretizes continuous motion into a sequence of discrete tokens from a learned codebook
# 2. **GPT-style Transformer**: Autoregressively generates motion tokens conditioned on text
# 3. **VQ-VAE Decoder**: Decodes predicted tokens back to continuous motion
# 
# **Key Properties:**
# - **Motion Tokenization**: Continuous motion `(B, T, F)` → discrete tokens `(B, T')` via temporal downsampling
# - **Autoregressive Generation**: $p(m_t | m_{<t}, c)$ where $m_t$ are motion tokens and $c$ is text condition
# - **CLIP Text Conditioning**: Text embeddings prepended as prefix tokens
# - **Classifier-free Guidance**: Random condition dropping during training
# 
# **Feature breakdown (263 dims for HumanML3D with 22 joints):**
# - Root data: 4 dims (1 rotation velocity + 2 linear velocity XZ + 1 height Y)
# - RIC data: (22-1) × 3 = 63 dims (root-invariant coordinates for non-root joints)
# - Rotation data: (22-1) × 6 = 126 dims (6D continuous rotation representation)
# - Local velocity: 22 × 3 = 66 dims (per-joint velocity)
# - Foot contact: 4 dims (binary contact labels for feet)
# 
# > **Training Strategy:**
# > 1. **Stage 1**: Train VQ-VAE to reconstruct motion (learns motion codebook)
# > 2. **Stage 2**: Train GPT on discrete motion tokens conditioned on text

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

# OpenAI CLIP (same import style as your existing notebook)
import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# %%

# -----------------------------
# Config
# -----------------------------

@dataclass
class TrainConfig:
    root: str = 'humanml'
    split: str = 'train'
    max_len: int = 196
    batch_size: int = 32
    num_workers: int = 0

    lr: float = 1e-4
    vqvae_steps: int = 100_000   # Stage 1: VQ-VAE training
    gpt_steps: int = 200_000     # Stage 2: GPT training
    grad_clip: float = 1.0

    # VQ-VAE
    codebook_size: int = 512     # Number of discrete codes
    codebook_dim: int = 512      # Dimension of each code
    commitment_cost: float = 0.25
    downsample_rate: int = 4     # Temporal downsampling factor

    # Classifier-free guidance
    p_uncond: float = 0.1
    guidance_scale: float = 2.5  # used at sampling

    # Model (GPT)
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1

cfg = TrainConfig()


# %%

# -----------------------------
# Load dataset normalization stats (provided by the KIT/HumanML3D publishers)
# -----------------------------

mean_path = os.path.join(cfg.root, 'Mean.npy')
std_path  = os.path.join(cfg.root, 'Std.npy')

mean = torch.from_numpy(np.load(mean_path)).float()
std  = torch.from_numpy(np.load(std_path)).float()

# Some releases store mean/std as (F,) and others as (1,F)
mean = mean.view(-1)
std  = std.view(-1)

Fdim = mean.shape[0]
print('feature dim (from Mean.npy):', Fdim)


# %%

# -----------------------------
# Dataset: HumanML3D new_joint_vecs (T, 263) + texts + CLIP embeddings
# -----------------------------

class HUMANML3DJointVecDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        max_len: Optional[int] = None,
        normalize: bool = True,
        use_cache: bool = True,
        text_encoder: Optional[callable] = None,  # Pass encode_text function
    ):
        self.root = root
        self.split = split
        self.max_len = max_len
        self.normalize = normalize
        self.text_encoder = text_encoder

        self.motion_dir = os.path.join(root, 'new_joint_vecs')
        self.text_dir = os.path.join(root, 'texts')
        self.split_file = os.path.join(root, f'{split}.txt')

        if not os.path.exists(self.motion_dir):
            raise FileNotFoundError(f"Missing {self.motion_dir}. Expected HUMAN-ML3D preprocessed folder 'new_joint_vecs'.")

        with open(self.split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # Cache paths
        motion_cache = f"HUMANML3D_joint_vecs_{split}_maxlen{max_len}_norm{normalize}.pt"
        self.motion_cache_path = os.path.join(root, motion_cache)
        self.emb_cache_path = os.path.join(root, f'clip_embeddings_{split}.pt')

        self.data: List[Dict] = []
        self.embeddings: Dict[str, torch.Tensor] = {}  # id -> (num_captions, 512) on GPU

        # Load or build motion data
        if use_cache and os.path.exists(self.motion_cache_path):
            self.data = torch.load(self.motion_cache_path)
        else:
            self._build_motion_cache()
            if use_cache:
                torch.save(self.data, self.motion_cache_path)

        # Load or build text embeddings (stored on GPU for fast training)
        if use_cache and os.path.exists(self.emb_cache_path) and text_encoder is not None:
            print(f'Loading cached CLIP embeddings from {self.emb_cache_path}')
            cpu_cache = torch.load(self.emb_cache_path)
            for mid, emb in cpu_cache.items():
                self.embeddings[mid] = emb.to(device)
            print(f'Loaded {len(self.embeddings)} embeddings to {device}')
        elif text_encoder is not None:
            self._build_embedding_cache()
            if use_cache:
                # Save CPU copies to disk
                cpu_cache = {k: v.cpu() for k, v in self.embeddings.items()}
                torch.save(cpu_cache, self.emb_cache_path)
                print(f'Saved embeddings to {self.emb_cache_path}')

    def _build_motion_cache(self):
        print('Building motion cache...')
        self.data = []
        for mid in self.ids:
            mpath = os.path.join(self.motion_dir, f'{mid}.npy')
            if not os.path.exists(mpath):
                continue
            motion = np.load(mpath).astype(np.float32)  # (T, F)

            if motion.ndim != 2:
                raise ValueError(f"Expected (T,F) for {mid}, got {motion.shape}")

            if self.max_len is not None:
                motion = motion[:self.max_len]

            # Load texts
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
            embs = self.text_encoder(texts, normalize=True)  # (num_captions, 512) on GPU
            self.embeddings[mid] = embs
        print(f'Computed {len(self.embeddings)} embeddings on {device}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item['motion'].clone()  # (T,F)

        if self.normalize:
            motion = (motion - mean) / (std + 1e-8)

        # Random caption index
        text_idx = random.randint(0, len(item['texts']) - 1)

        return {
            'id': item['id'],
            'motion': motion,
            'length': item['length'],
            'text_idx': text_idx,
        }

    def get_embedding(self, sample_id: str, text_idx: int) -> torch.Tensor:
        """Get pre-computed embedding for a sample. Returns tensor on GPU."""
        return self.embeddings[sample_id][text_idx]


def collate_joint_vecs(batch: List[Dict]):
    motions = [b['motion'] for b in batch]
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    ids = [b['id'] for b in batch]
    text_idxs = [b['text_idx'] for b in batch]

    B = len(motions)
    T_max = int(lengths.max())
    Fdim = motions[0].shape[1]

    x = torch.zeros(B, T_max, Fdim, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, m in enumerate(motions):
        T = m.shape[0]
        x[i, :T] = m
        mask[i, :T] = True

    return {
        'motion': x,
        'mask': mask,
        'lengths': lengths,
        'ids': ids,
        'text_idxs': text_idxs,
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
    """Encode texts with CLIP. Optionally L2-normalize (recommended for diffusion)."""
    tokens = clip.tokenize(texts, truncate=True).to(device)
    emb = clip_model.encode_text(tokens).float()  # (B, 512)
    if normalize:
        emb = F.normalize(emb, dim=-1)
    return emb

# Pre-compute the "unconditional" embedding (empty string) once - lives on GPU
empty_emb = encode_text(['']).squeeze(0)  # (512,) on GPU
print('Empty embedding norm:', empty_emb.norm().item())


# %%

# -----------------------------
# Create dataset with embedded text (pass encoder for caching)
# -----------------------------

dataset = HUMANML3DJointVecDataset(
    cfg.root, 
    split='train', 
    max_len=cfg.max_len, 
    normalize=True, 
    use_cache=True,
    text_encoder=encode_text,  # Pass encoder so dataset can compute/cache embeddings
)
print('dataset size:', len(dataset))

loader = DataLoader(
    dataset, 
    batch_size=cfg.batch_size, 
    shuffle=True, 
    num_workers=cfg.num_workers, 
    collate_fn=collate_joint_vecs, 
    drop_last=True
)

batch = next(iter(loader))
print('x:', batch['motion'].shape, 'mask:', batch['mask'].shape)


# %%

# -----------------------------
# VQ-VAE: Vector Quantization Variational Autoencoder for Motion Tokenization
# -----------------------------

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer using exponential moving average (EMA) codebook updates.
    Converts continuous latent vectors to discrete codes.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Codebook: learnable embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA cluster statistics
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Continuous latent (B, T', D) where T' is downsampled time
        Returns:
            z_q: Quantized latent (B, T', D)
            indices: Codebook indices (B, T')
            vq_loss: VQ commitment + codebook loss
        """
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*T', D)

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )  # (B*T', K)

        # Find nearest codebook entry
        indices = d.argmin(dim=1)  # (B*T',)
        z_q = self.embedding(indices).view(B, T, D)  # (B, T', D)

        # VQ losses
        if self.training:
            # EMA update for codebook
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_embeddings).float()  # (B*T', K)
                self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
                dw = z_flat.t() @ encodings  # (D, K)
                self.ema_w.mul_(self.decay).add_(dw.t(), alpha=1 - self.decay)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # Commitment loss: encoder should commit to codebook
        commitment_loss = F.mse_loss(z, z_q.detach())

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()

        indices = indices.view(B, T)
        return z_q, indices, self.commitment_cost * commitment_loss


class MotionVQVAE(nn.Module):
    """
    VQ-VAE for motion tokenization (MotionGPT-style).
    Encoder: 1D Conv with temporal downsampling
    Decoder: 1D Conv with temporal upsampling
    """
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

        # Encoder: motion -> latent (with temporal downsampling)
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.Conv1d(512, codebook_dim, kernel_size=3, padding=1),
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)

        # Decoder: latent -> motion (with temporal upsampling)
        self.decoder = nn.Sequential(
            nn.Conv1d(codebook_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1),  # *2
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),  # *4
            nn.ReLU(inplace=True),
            nn.Conv1d(256, feature_dim, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode motion to discrete tokens.
        Args:
            x: Motion (B, T, F)
            mask: Valid frame mask (B, T)
        Returns:
            indices: Codebook indices (B, T')
            z_q: Quantized latent (B, T', D)
        """
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)  # (B, F, T)
        z = self.encoder(x)    # (B, D, T')
        z = z.transpose(1, 2)  # (B, T', D)

        z_q, indices, _ = self.vq(z)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent to motion.
        Args:
            z_q: Quantized latent (B, T', D)
        Returns:
            x_recon: Reconstructed motion (B, T, F)
        """
        z_q = z_q.transpose(1, 2)  # (B, D, T')
        x = self.decoder(z_q)      # (B, F, T)
        x = x.transpose(1, 2)      # (B, T, F)
        return x

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from codebook indices directly.
        Args:
            indices: Codebook indices (B, T')
        Returns:
            x_recon: Reconstructed motion (B, T, F)
        """
        z_q = self.vq.embedding(indices)  # (B, T', D)
        return self.decode(z_q)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.
        Args:
            x: Motion (B, T, F)
            mask: Valid frame mask (B, T)
        Returns:
            x_recon: Reconstructed motion (B, T, F)
            indices: Codebook indices (B, T')
            vq_loss: VQ commitment loss
        """
        # Encoder
        x_conv = x.transpose(1, 2)  # (B, F, T)
        z = self.encoder(x_conv)     # (B, D, T')
        z = z.transpose(1, 2)        # (B, T', D)

        # Quantize
        z_q, indices, vq_loss = self.vq(z)

        # Decoder
        x_recon = self.decode(z_q)

        return x_recon, indices, vq_loss


# Create VQ-VAE model
vqvae = MotionVQVAE(
    feature_dim=Fdim,
    codebook_size=cfg.codebook_size,
    codebook_dim=cfg.codebook_dim,
    downsample_rate=cfg.downsample_rate,
    commitment_cost=cfg.commitment_cost,
).to(device)

n_params = sum(p.numel() for p in vqvae.parameters())
print(f'VQ-VAE params: {n_params/1e6:.2f}M')


# %%

# -----------------------------
# Stage 1: VQ-VAE Training (Motion Reconstruction)
# -----------------------------

def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss with mask for variable-length sequences."""
    # a, b: (B, T, F), mask: (B, T)
    m = mask.float().unsqueeze(-1)
    se = (a - b) ** 2
    se = se * m
    denom = m.sum() * se.shape[-1] + 1e-8
    return se.sum() / denom


def vel_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Velocity consistency loss for smooth motion reconstruction."""
    m = mask[:, 1:] & mask[:, :-1]
    v_gt = x[:, 1:] - x[:, :-1]
    v_pr = x_hat[:, 1:] - x_hat[:, :-1]
    return masked_mse(v_gt, v_pr, m)


vqvae_optimizer = torch.optim.AdamW(vqvae.parameters(), lr=cfg.lr)

# Reconstruction loss weights
lambda_recon = 1.0
lambda_vel = 0.5  # velocity smoothness


# %%

# -----------------------------
# VQ-VAE Training Loop
# -----------------------------

from itertools import cycle

vqvae.train()
data_iter = cycle(loader)

print("Stage 1: Training VQ-VAE for motion tokenization...")

for step in range(1, cfg.vqvae_steps + 1):
    batch = next(data_iter)
    x = batch['motion'].to(device)      # (B, T, Fdim) normalized
    mask = batch['mask'].to(device)     # (B, T)

    # Pad to multiple of downsample_rate for clean conv/deconv
    B, T, _ = x.shape
    pad_len = (cfg.downsample_rate - T % cfg.downsample_rate) % cfg.downsample_rate
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        mask = F.pad(mask, (0, pad_len), value=False)

    # Forward pass
    x_recon, indices, vq_loss = vqvae(x, mask)

    # Truncate back to original length for loss computation
    x_recon = x_recon[:, :T]
    mask_orig = batch['mask'].to(device)

    # Reconstruction loss
    recon_loss = masked_mse(batch['motion'].to(device), x_recon, mask_orig)

    # Velocity loss for smoothness
    v_loss = vel_loss(batch['motion'].to(device), x_recon, mask_orig)

    # Total loss
    loss = lambda_recon * recon_loss + vq_loss + lambda_vel * v_loss

    vqvae_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(vqvae.parameters(), cfg.grad_clip)
    vqvae_optimizer.step()

    if step % 200 == 0:
        print(f"step {step:>7d} | recon {recon_loss.item():.5f} | vq {vq_loss.item():.5f} | vel {v_loss.item():.5f}")

    # Save checkpoints
    if step % 10_000 == 0:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model': vqvae.state_dict(),
            'optimizer': vqvae_optimizer.state_dict(),
            'step': step,
            'cfg': cfg.__dict__,
        }, f'checkpoints/vqvae_humanml3d_step{step}.pt')
        print('saved VQ-VAE checkpoint')

print("VQ-VAE training complete!")


# %%

# -----------------------------
# GPT Model: Autoregressive Transformer for Motion Token Generation
# -----------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D/H)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T) for key masking
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class GPTBlock(nn.Module):
    """Transformer block with causal self-attention."""
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
    """
    GPT-style autoregressive model for motion token generation.
    
    Sequence format: [COND] [BOS] m_1 m_2 ... m_T [EOS]
    - COND: Text condition embedding projected to d_model
    - BOS: Beginning of sequence token
    - m_i: Motion tokens from VQ-VAE codebook
    - EOS: End of sequence token
    """
    def __init__(
        self,
        codebook_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 256,
        cond_dim: int = 512,  # CLIP embedding dimension
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.max_len = max_len

        # Special tokens: BOS, EOS, PAD
        self.bos_token = codebook_size
        self.eos_token = codebook_size + 1
        self.pad_token = codebook_size + 2
        self.vocab_size = codebook_size + 3

        # Token embeddings
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Condition projection (CLIP -> d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.head.weight

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        Args:
            tokens: Motion token indices (B, T) including BOS, excluding EOS in input
            cond: CLIP text embedding (B, cond_dim)
            mask: Valid token mask (B, T)
        Returns:
            logits: (B, T, vocab_size) for next token prediction
        """
        B, T = tokens.shape

        # Token + position embeddings
        tok_emb = self.token_emb(tokens)  # (B, T, d)
        pos_emb = self.pos_emb[:, :T, :]
        h = tok_emb + pos_emb

        # Add condition as prefix token
        cond_token = self.cond_proj(cond).unsqueeze(1)  # (B, 1, d)
        h = torch.cat([cond_token, h], dim=1)  # (B, 1+T, d)

        # Update mask for condition token
        if mask is not None:
            mask = torch.cat([
                torch.ones(B, 1, dtype=torch.bool, device=mask.device),
                mask
            ], dim=1)

        # Transformer
        for block in self.blocks:
            h = block(h, mask)

        h = self.ln_f(h)
        logits = self.head(h)  # (B, 1+T, vocab_size)

        # Remove condition token position from logits
        logits = logits[:, 1:, :]  # (B, T, vocab_size)

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
        """
        Autoregressive generation with classifier-free guidance.
        Args:
            cond: CLIP text embedding (B, cond_dim)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            guidance_scale: Classifier-free guidance strength
            cond_uncond: Unconditional embedding for CFG
        Returns:
            tokens: Generated motion tokens (B, T') without BOS/EOS
        """
        B = cond.shape[0]
        device = cond.device

        # Start with BOS token
        tokens = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self(tokens, cond)[:, -1, :]  # (B, vocab_size)

            # Classifier-free guidance
            if guidance_scale > 0 and cond_uncond is not None:
                logits_uncond = self(tokens, cond_uncond)[:, -1, :]
                logits = logits_uncond + guidance_scale * (logits - logits_uncond)

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if all sequences have EOS
            if (next_token == self.eos_token).all():
                break

        # Remove BOS and EOS tokens
        result = []
        for i in range(B):
            seq = tokens[i, 1:]  # Remove BOS
            eos_pos = (seq == self.eos_token).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                seq = seq[:eos_pos[0]]  # Remove EOS and after
            # Filter out any special tokens
            seq = seq[seq < self.codebook_size]
            result.append(seq)

        # Pad to same length
        max_len = max(len(s) for s in result) if result else 0
        padded = torch.full((B, max_len), self.pad_token, dtype=torch.long, device=device)
        for i, seq in enumerate(result):
            if len(seq) > 0:
                padded[i, :len(seq)] = seq

        return padded


# Create GPT model
gpt = MotionGPT(
    codebook_size=cfg.codebook_size,
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    dropout=cfg.dropout,
    max_len=cfg.max_len // cfg.downsample_rate + 10,  # Account for downsampling + special tokens
    cond_dim=512,  # CLIP dimension
).to(device)

n_params = sum(p.numel() for p in gpt.parameters())
print(f'MotionGPT params: {n_params/1e6:.2f}M')


# %%

# -----------------------------
# Stage 2: GPT Training (Autoregressive Motion Token Generation)
# -----------------------------

def prepare_gpt_batch(batch, vqvae, downsample_rate):
    """
    Prepare batch for GPT training.
    Encodes motion to tokens and creates input/target pairs.
    """
    x = batch['motion'].to(device)
    mask = batch['mask'].to(device)
    B, T, F = x.shape

    # Pad to multiple of downsample_rate
    pad_len = (downsample_rate - T % downsample_rate) % downsample_rate
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))

    # Encode to tokens
    with torch.no_grad():
        indices, _ = vqvae.encode(x)  # (B, T')

    T_tokens = indices.shape[1]

    # Compute token mask from original mask
    # Each token covers `downsample_rate` frames
    token_lengths = (batch['lengths'].to(device) + downsample_rate - 1) // downsample_rate
    token_mask = torch.arange(T_tokens, device=device).unsqueeze(0) < token_lengths.unsqueeze(1)

    # Create input: [BOS] m_1 m_2 ... m_{T-1}
    # Create target: m_1 m_2 ... m_T [EOS]
    bos = torch.full((B, 1), gpt.bos_token, dtype=torch.long, device=device)
    eos = torch.full((B, 1), gpt.eos_token, dtype=torch.long, device=device)

    # Input sequence: BOS + motion tokens (without last one for teacher forcing)
    input_tokens = torch.cat([bos, indices], dim=1)  # (B, 1+T')

    # Target sequence: motion tokens + EOS
    target_tokens = torch.cat([indices, eos], dim=1)  # (B, T'+1)

    # Mask for loss computation (include EOS position)
    target_mask = torch.cat([token_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)

    return input_tokens, target_tokens, target_mask


# %%

# -----------------------------
# GPT Training Loop
# -----------------------------

# Load trained VQ-VAE (or use the one just trained)
# vqvae_checkpoint = torch.load('checkpoints/vqvae_humanml3d_step100000.pt', map_location=device)
# vqvae.load_state_dict(vqvae_checkpoint['model'])

vqvae.eval()  # Freeze VQ-VAE during GPT training

gpt_optimizer = torch.optim.AdamW(gpt.parameters(), lr=cfg.lr)

gpt.train()
data_iter = cycle(loader)

print("Stage 2: Training MotionGPT for autoregressive generation...")

for step in range(1, cfg.gpt_steps + 1):
    batch = next(data_iter)

    # Prepare GPT training data
    input_tokens, target_tokens, target_mask = prepare_gpt_batch(batch, vqvae, cfg.downsample_rate)

    # Get text embeddings (with classifier-free guidance dropout)
    cond_list = []
    for mid, tidx in zip(batch['ids'], batch['text_idxs']):
        if random.random() < cfg.p_uncond:
            cond_list.append(empty_emb)  # unconditional
        else:
            cond_list.append(dataset.get_embedding(mid, tidx))
    cond = torch.stack(cond_list)  # (B, 512)

    # Forward pass
    logits = gpt(input_tokens, cond)  # (B, T, vocab_size)

    # Cross-entropy loss with mask
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        target_tokens.reshape(-1),
        ignore_index=gpt.pad_token,
        reduction='none'
    ).reshape(B, T)

    # Apply mask
    loss = (loss * target_mask.float()).sum() / (target_mask.float().sum() + 1e-8)

    gpt_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(gpt.parameters(), cfg.grad_clip)
    gpt_optimizer.step()

    if step % 200 == 0:
        # Compute perplexity
        ppl = torch.exp(loss).item()
        print(f"step {step:>7d} | loss {loss.item():.5f} | ppl {ppl:.2f}")

    # Save checkpoints
    if step % 10_000 == 0:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'gpt': gpt.state_dict(),
            'vqvae': vqvae.state_dict(),
            'optimizer': gpt_optimizer.state_dict(),
            'step': step,
            'cfg': cfg.__dict__,
        }, f'checkpoints/motiongpt_humanml3d_step{step}.pt')
        print('saved MotionGPT checkpoint')

print("MotionGPT training complete!")


# %%

# -----------------------------
# Sampling: Text-to-Motion Generation with Classifier-Free Guidance
# -----------------------------

# Load checkpoint (uncomment to load from file)
# checkpoint = torch.load('checkpoints/motiongpt_humanml3d_step200000.pt', map_location=device)
# gpt.load_state_dict(checkpoint['gpt'])
# vqvae.load_state_dict(checkpoint['vqvae'])

@torch.no_grad()
def sample_text(
    prompt: str,
    length: int = 196,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    guidance_scale: float = 2.5,
):
    """
    Generate motion from text prompt using MotionGPT.
    
    Args:
        prompt: Text description of motion
        length: Target motion length in frames
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
        guidance_scale: Classifier-free guidance strength
        
    Returns:
        motion: Generated motion (T, F) in original feature space
    """
    gpt.eval()
    vqvae.eval()

    # Encode text condition
    cond = encode_text([prompt])  # (1, 512)
    cond_uncond = encode_text([''])  # (1, 512)

    # Calculate target token length
    max_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate

    # Generate motion tokens autoregressively
    tokens = gpt.generate(
        cond=cond,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        guidance_scale=guidance_scale,
        cond_uncond=cond_uncond,
    )  # (1, T')

    # Decode tokens to motion
    motion_norm = vqvae.decode_indices(tokens)  # (1, T_upsampled, F)

    # Truncate to target length
    motion_norm = motion_norm[0, :length]  # (T, F)

    # Unnormalize
    motion = motion_norm.cpu() * (std + 1e-8) + mean

    return motion


# Example generation
out_text = 'a person does a kick'
out = sample_text(out_text, length=100, guidance_scale=cfg.guidance_scale)
print('Generated motion shape:', out.shape)

# Save output
out_np = out.detach().cpu().numpy()
np.save("sample_output_gpt.npy", out_np)
print(f"Saved to sample_output_gpt.npy")


# %%

# -----------------------------
# Visualization: Animate Generated Motion
# -----------------------------

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# --- HumanML3D skeleton connectivity (22 joints, SMPL-based) ---
HUMANML3D_EDGES = [
    # Left leg: 0 -> 2 -> 5 -> 8 -> 11
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Right leg: 0 -> 1 -> 4 -> 7 -> 10
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Spine/head: 0 -> 3 -> 6 -> 9 -> 12 -> 15
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left arm: 9 -> 14 -> 17 -> 19 -> 21
    (9, 14), (14, 17), (17, 19), (19, 21),
    # Right arm: 9 -> 13 -> 16 -> 18 -> 20
    (9, 13), (13, 16), (16, 18), (18, 20),
]

def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """
    Recover 3D joint positions from HumanML3D/KIT-ML new_joint_vecs representation.
    """
    data = np.asarray(data, dtype=np.float32)
    T = data.shape[0]
    
    # --- Recover root rotation (yaw around Y-axis) ---
    r_rot_vel = data[:, 0]  # (T,)
    r_rot_ang = np.zeros(T, dtype=np.float32)
    r_rot_ang[1:] = np.cumsum(r_rot_vel[:-1])
    
    # --- Recover root position (world coordinates) ---
    r_pos = np.zeros((T, 3), dtype=np.float32)
    r_pos[:, 1] = data[:, 3]  # Y is absolute height
    
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
    
    # --- Extract RIC (root-invariant coordinates) ---
    ric = data[:, 4:4 + (joints_num - 1) * 3]
    ric = ric.reshape(T, joints_num - 1, 3)
    
    # Rotate RIC from root-local frame to world frame
    positions = np.zeros((T, joints_num - 1, 3), dtype=np.float32)
    positions[:, :, 0] = cos_r[:, None] * ric[:, :, 0] - sin_r[:, None] * ric[:, :, 2]
    positions[:, :, 1] = ric[:, :, 1]
    positions[:, :, 2] = sin_r[:, None] * ric[:, :, 0] + cos_r[:, None] * ric[:, :, 2]
    
    positions[:, :, 0] += r_pos[:, 0:1]
    positions[:, :, 2] += r_pos[:, 2:3]
    
    joints = np.concatenate([r_pos[:, None, :], positions], axis=1)
    return joints

def animate_skeleton(motion, edges=HUMANML3D_EDGES, title="Motion", stride=1, elev=15, azim=-70, interval=80, center=True):
    """Animate a skeleton from joint positions (T, J, 3)."""
    motion = motion[::stride].copy()
    T, J, _ = motion.shape
    
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
        
        for k, (i, j) in enumerate(edges):
            lines[k].set_data([xs[i], xs[j]], [ys[i], ys[j]])
            lines[k].set_3d_properties([zs[i], zs[j]])
        
        ax.set_title(f"{title} | frame {t+1}/{T}")
        return [pts] + lines

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval, blit=False)
    plt.close(fig)
    return anim


# Visualize generated motion
vecs = np.load("sample_output_gpt.npy")
joints = recover_from_ric(vecs, joints_num=22)
print("Joints shape:", joints.shape, "range:", joints.min(), joints.max())

anim = animate_skeleton(joints, edges=HUMANML3D_EDGES, title=f"MotionGPT: {out_text}", center=True)
display(HTML(anim.to_jshtml()))



