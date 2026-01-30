# %% [markdown]
# # MDM-style Diffusion on HumanML3D `new_joint_vecs` (T, 263)
# 
# This notebook implements the core training setup described in **Motion Diffusion Model (MDM)** (Tevet et al., 2022) on the **HumanML3D** dataset using the **redundant per-frame feature representation** (`new_joint_vecs`: `(frames, features)` where `features=263`).
# 
# **HumanML3D vs KIT-ML Feature Dimensions:**
# - **HumanML3D**: 22 joints → 263 dimensions
# - **KIT-ML**: 21 joints → 251 dimensions
# 
# **Feature breakdown (263 dims for HumanML3D with 22 joints):**
# - Root data: 4 dims (1 rotation velocity + 2 linear velocity XZ + 1 height Y)
# - RIC data: (22-1) × 3 = 63 dims (root-invariant coordinates for non-root joints)
# - Rotation data: (22-1) × 6 = 126 dims (6D continuous rotation representation)
# - Local velocity: 22 × 3 = 66 dims (per-joint velocity)
# - Foot contact: 4 dims (binary contact labels for feet)
# 
# Key properties (matching the paper):
# - **Diffusion over motion vectors** `x_t` (shape `(B, T, F)`)
# - **Model predicts the clean sample** `x̂₀ = G(x_t, t, c)` (x0-prediction)
# - **Transformer encoder-only backbone**
# - **Classifier-free guidance** via random condition dropping during training
# 
# > Notes:
# > - For KIT/HumanML3D, the paper *does not* use geometric losses in the text-to-motion experiments because joint locations and foot-contact are already explicitly represented.
# > - You can optionally enable additional regularizers (velocity / contact) if desired.

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
    steps: int = 200_000
    grad_clip: float = 1.0

    # Diffusion
    T: int = 1000

    # Classifier-free guidance
    p_uncond: float = 0.1
    guidance_scale: float = 2.5  # used at sampling

    # Model
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
# Diffusion helpers (cosine schedule) + x0-prediction sampling
# -----------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    # From Nichol & Dhariwal, improved DDPM
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

class Diffusion:
    def __init__(self, T: int):
        self.T = T
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register(betas=betas, alphas=alphas, alpha_bar=alpha_bar)

    def register(self, **tensors):
        for k,v in tensors.items():
            setattr(self, k, v.to(device))

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x0: (B,T,F)
        B = x0.shape[0]
        shape = [B] + [1]*(x0.ndim-1)
        s1 = self.sqrt_alpha_bar[t].view(*shape)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(*shape)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model, xt: torch.Tensor, t: int, cond: torch.Tensor, mask: torch.Tensor,
                 guidance_scale: float = 0.0, cond_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Predict x0, then sample xt-1 using DDPM posterior
        B = xt.shape[0]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        if guidance_scale > 0 and cond_uncond is not None:
            x0_uncond = model(xt, t_batch, cond_uncond, mask)
            x0_cond   = model(xt, t_batch, cond, mask)
            x0_hat = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        else:
            x0_hat = model(xt, t_batch, cond, mask)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        # Eq. for mean in terms of x0_hat and xt
        coef1 = torch.sqrt(self.alpha_bar[t-1]) * beta_t / (1 - alpha_bar_t) if t > 0 else 0.0
        coef2 = torch.sqrt(alpha_t) * (1 - self.alpha_bar[t-1]) / (1 - alpha_bar_t) if t > 0 else 0.0

        if t == 0:
            x_prev = x0_hat
        else:
            mean = coef1 * x0_hat + coef2 * xt
            var = beta_t * (1 - self.alpha_bar[t-1]) / (1 - alpha_bar_t)
            noise = torch.randn_like(xt)
            x_prev = mean + torch.sqrt(var) * noise

        # keep padding stable
        x_prev = x_prev * mask.float().unsqueeze(-1)
        return x_prev

    @torch.no_grad()
    def sample(self, model, shape: Tuple[int,int,int], cond: torch.Tensor, mask: torch.Tensor,
               guidance_scale: float = 2.5, cond_uncond: Optional[torch.Tensor] = None):
        x = torch.randn(shape, device=device)
        x = x * mask.float().unsqueeze(-1)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond, mask, guidance_scale=guidance_scale, cond_uncond=cond_uncond)
        return x


diff = Diffusion(cfg.T)


# %%

# -----------------------------
# Model: Transformer encoder-only predicting x0 (as in Fig.2 of MDM)
# -----------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    # Standard sinusoidal embedding
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class MDMTransformer(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int = 512, d_model: int = 512, n_layers: int = 8, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        self.frame_in = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model))  # enough for HumanML3D lengths; slice at runtime

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.c_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor):
        # xt: (B,T,F), cond: (B,512), mask: (B,T)
        B, T, Fdim = xt.shape

        # (B,T,d)
        h = self.frame_in(xt)
        h = h + self.pos_emb[:, :T, :]

        # z_tk token: time + condition
        t_emb = timestep_embedding(t, self.d_model)
        zt = self.t_mlp(t_emb) + self.c_mlp(cond)
        zt = zt.unsqueeze(1)  # (B,1,d)

        # prepend token
        tokens = torch.cat([zt, h], dim=1)  # (B,1+T,d)

        # transformer padding mask: True means "ignore" in torch
        src_key_padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
            ~mask
        ], dim=1)

        y = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)  # (B,1+T,d)
        y = y[:, 1:, :]  # drop token

        x0_hat = self.out(y)
        x0_hat = x0_hat * mask.float().unsqueeze(-1)
        return x0_hat

model = MDMTransformer(
    feature_dim=Fdim,
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    dropout=cfg.dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print('params:', n_params/1e6, 'M')


# %%

# -----------------------------
# Training step (MDM objective): L_simple + optional regularizers
# -----------------------------

def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # a,b: (B,T,F), mask:(B,T)
    m = mask.float().unsqueeze(-1)
    se = (a - b) ** 2
    se = se * m
    denom = m.sum() * se.shape[-1] + 1e-8
    return se.sum() / denom

# Optional: velocity regularizer (feature-space). For KIT joint-vecs you may prefer using the true velocity channels.
# Here we implement a generic smoothness term on the predicted x0.

def vel_loss(x0: torch.Tensor, x0_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # compare finite differences
    m = mask[:, 1:] & mask[:, :-1]
    v_gt = x0[:, 1:] - x0[:, :-1]
    v_pr = x0_hat[:, 1:] - x0_hat[:, :-1]
    return masked_mse(v_gt, v_pr, m)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

# weights (match paper form; set to 0 to disable)
lambda_vel = 0.0



# %%

# -----------------------------
# Main training loop (minimal)
# -----------------------------

from itertools import cycle

model.train()

data_iter = cycle(loader)

for step in range(1, cfg.steps + 1):
    batch = next(data_iter)
    x0 = batch['motion'].to(device)          # (B,T,F) normalized
    mask = batch['mask'].to(device)          # (B,T)

    # Look up pre-computed embeddings from dataset (already on GPU!)
    cond_list = []
    for mid, tidx in zip(batch['ids'], batch['text_idxs']):
        if random.random() < cfg.p_uncond:
            cond_list.append(empty_emb)  # unconditional, on GPU
        else:
            cond_list.append(dataset.get_embedding(mid, tidx))  # on GPU
    cond = torch.stack(cond_list)  # (B, 512) - already on GPU, no transfer!

    B = x0.shape[0]
    t = torch.randint(0, cfg.T, (B,), device=device)
    noise = torch.randn_like(x0)
    xt = diff.q_sample(x0, t, noise)
    xt = xt * mask.float().unsqueeze(-1)

    x0_hat = model(xt, t, cond, mask)

    loss = masked_mse(x0, x0_hat, mask)
    if lambda_vel > 0:
        loss = loss + lambda_vel * vel_loss(x0, x0_hat, mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step:>7d} | loss {loss.item():.5f}")

    # Save lightweight checkpoints occasionally
    if step % 10_000 == 0:
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'cfg': cfg.__dict__,
        }, f'checkpoints/mdm_humanml3d_jointvecs_step{step}.pt')

        print('saved checkpoint')


# %%

# -----------------------------
# Sampling (classifier-free guidance)
# -----------------------------
checkpoint = torch.load('checkpoints/mdm_kit_jointvecs_step200000.pt', map_location=device)

# Create a fresh model instance and load the weights
model = MDMTransformer(
    feature_dim=Fdim,
    d_model=cfg.d_model,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads,
    dropout=cfg.dropout,
).to(device)
model.load_state_dict(checkpoint['model'])

@torch.no_grad()
def sample_text(prompt: str, length: int = 196, guidance_scale: float = 2.5):
    model.eval()

    B = 1
    T = length
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    cond = encode_text([prompt])
    cond_uncond = encode_text([''])

    x_norm = diff.sample(model, (B, T, Fdim), cond=cond, cond_uncond=cond_uncond, mask=mask, guidance_scale=guidance_scale)

    # unnormalize back to original joint-vec scale
    x = x_norm[0].cpu() * (std + 1e-8) + mean
    return x  # (T,F)

out_text = 'a person does a kick'
# Example (after training for a while):
out = sample_text(out_text, length=100, guidance_scale=cfg.guidance_scale)
print(out.shape)
out_np = out.detach().cpu().numpy()
np.save("sample_output.npy", out_np)


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# --- HumanML3D skeleton connectivity (22 joints, SMPL-based) ---
# Based on t2m_kinematic_chain from paramUtil.py:
# [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
# Joint indices:
#  0: pelvis (root)
#  1: right_hip, 2: left_hip, 3: spine1
#  4: right_knee, 5: left_knee, 6: spine2
#  7: right_ankle, 8: left_ankle, 9: spine3
# 10: right_foot, 11: left_foot, 12: neck
# 13: right_collar, 14: left_collar, 15: head
# 16: right_shoulder, 17: left_shoulder
# 18: right_elbow, 19: left_elbow
# 20: right_wrist, 21: left_wrist

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
    
    NOTE: This returns WORLD COORDINATES - no centering is applied here.
    The diffusion model operates on the original data representation, and this
    function faithfully reconstructs the world-space motion trajectory.
    
    Feature layout (for joints_num joints):
    - [0]: root angular velocity (around Y)
    - [1:3]: root linear velocity (XZ plane, in local frame)
    - [3]: root height (Y)
    - [4 : 4 + (joints_num-1)*3]: RIC (root-invariant coordinates) for non-root joints
    - ... (remaining features: rotations, local velocities, foot contacts)
    
    Args:
        data: Motion features, shape (T, F) where F = 4 + (j-1)*3 + (j-1)*6 + j*3 + 4
        joints_num: Number of joints (22 for HumanML3D, 21 for KIT-ML)
    
    Returns:
        joints: Shape (T, joints_num, 3) - 3D positions in WORLD frame (not centered)
    """
    data = np.asarray(data, dtype=np.float32)
    T = data.shape[0]
    
    # --- Recover root rotation (yaw around Y-axis) ---
    r_rot_vel = data[:, 0]  # (T,)
    r_rot_ang = np.zeros(T, dtype=np.float32)
    r_rot_ang[1:] = np.cumsum(r_rot_vel[:-1])  # integrate angular velocity
    
    # --- Recover root position (world coordinates) ---
    r_pos = np.zeros((T, 3), dtype=np.float32)
    r_pos[:, 1] = data[:, 3]  # Y is absolute height
    
    # X and Z come from integrating local velocities, rotated to world frame
    r_vel_local = np.zeros((T, 3), dtype=np.float32)
    r_vel_local[1:, 0] = data[:-1, 1]  # local X velocity
    r_vel_local[1:, 2] = data[:-1, 2]  # local Z velocity
    
    # Rotate local velocity to world frame using yaw angle
    cos_r = np.cos(r_rot_ang)
    sin_r = np.sin(r_rot_ang)
    r_vel_world = np.zeros_like(r_vel_local)
    r_vel_world[:, 0] = cos_r * r_vel_local[:, 0] - sin_r * r_vel_local[:, 2]
    r_vel_world[:, 2] = sin_r * r_vel_local[:, 0] + cos_r * r_vel_local[:, 2]
    
    r_pos[:, 0] = np.cumsum(r_vel_world[:, 0])
    r_pos[:, 2] = np.cumsum(r_vel_world[:, 2])
    
    # --- Extract RIC (root-invariant coordinates) for non-root joints ---
    ric = data[:, 4:4 + (joints_num - 1) * 3]  # (T, (joints_num-1)*3)
    ric = ric.reshape(T, joints_num - 1, 3)     # (T, joints_num-1, 3)
    
    # Rotate RIC from root-local frame to world frame
    positions = np.zeros((T, joints_num - 1, 3), dtype=np.float32)
    positions[:, :, 0] = cos_r[:, None] * ric[:, :, 0] - sin_r[:, None] * ric[:, :, 2]
    positions[:, :, 1] = ric[:, :, 1]  # Y stays the same (already world-aligned)
    positions[:, :, 2] = sin_r[:, None] * ric[:, :, 0] + cos_r[:, None] * ric[:, :, 2]
    
    # Add root XZ translation (Y is already absolute in RIC)
    positions[:, :, 0] += r_pos[:, 0:1]
    positions[:, :, 2] += r_pos[:, 2:3]
    
    # Concatenate root + other joints
    joints = np.concatenate([r_pos[:, None, :], positions], axis=1)  # (T, joints_num, 3)
    return joints

def animate_skeleton(motion, edges=HUMANML3D_EDGES, title="Motion", stride=1, elev=15, azim=-70, interval=80, center=True):
    """
    Animate a skeleton from joint positions (T, J, 3).
    
    Args:
        motion: Joint positions, shape (T, J, 3) - in world coordinates
        edges: List of (parent, child) tuples defining skeleton connectivity
        title: Title for the animation
        stride: Frame skip for faster playback
        elev, azim: Camera angles
        interval: Milliseconds between frames
        center: If True, center skeleton at root joint for better visualization.
                This is ONLY for display - does not affect the diffusion process.
                Set to False to see the actual world-space trajectory.
    """
    motion = motion[::stride].copy()
    T, J, _ = motion.shape
    
    # --- VISUALIZATION ONLY: optionally center on root joint ---
    # This does NOT affect the diffusion model - it only changes how we display the animation.
    # Set center=False to see the actual world-space motion trajectory.
    if center:
        motion = motion - motion[:, [0], :]
    
    # Compute bounds
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
        # Plot with Y-up: X, Z, Y mapping to display axes
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

# Load and visualize
vecs = np.load("sample_output.npy")  # (T, 263) for HumanML3D
joints = recover_from_ric(vecs, joints_num=22)  # (T, 22, 3) - WORLD coordinates, not centered
print("Joints shape:", joints.shape, "range:", joints.min(), joints.max())

# Visualize with centering enabled (for cleaner display)
# Set center=False to see the actual world-space trajectory
anim = animate_skeleton(joints, edges=HUMANML3D_EDGES, title=f"Generated: {out_text}", center=False)
display(HTML(anim.to_jshtml()))


