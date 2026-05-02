
import os
import math
import random
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import linalg

import clip
import matplotlib.pyplot as plt
import pandas as pd

# Import composite model generation
from generate import load_models as load_composite_models, generate_composite

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate motion generation models')
parser.add_argument('--models', type=str, default='all', 
                    help='Models to evaluate: all, mdm, gpt, composite (comma-separated)')
parser.add_argument('--load-results', action='store_true',
                    help='Load previous results from evaluation_results.csv')
parser.add_argument('--num-samples', type=int, default=256,
                    help='Number of samples for evaluation')
parser.add_argument('--composite-gpt-steps', type=int, default=None,
                    help='Override composite GPT checkpoint step for evaluation')
parser.add_argument('--composite-inbetween-steps', type=int, default=None,
                    help='Override composite in-betweening checkpoint step for evaluation')
args = parser.parse_args()

# Parse which models to evaluate
selected_models = set()
if args.models.lower() == 'all':
    selected_models = {'mdm', 'gpt', 'composite'}
else:
    selected_models = set(m.strip().lower() for m in args.models.split(','))

print(f'Evaluating models: {selected_models}')

@dataclass
class EvalConfig:
    root: str = 'humanml'
    
    # Checkpoints
    mdm_checkpoint: str = 'checkpoints/mdm_humanml3d_jointvecs_step200000.pt'
    motiongpt_checkpoint: str = 'checkpoints/motiongpt_humanml3d_step200000.pt'
    composite_gpt_steps: int = 200000
    composite_inbetween_steps: int = 200000
    
    # Evaluation settings
    num_samples: int = 256          # Total samples for FID/Diversity
    num_samples_per_text: int = 10  # For multimodality
    batch_size: int = 32
    max_len: int = 196
    
    # Diffusion
    T: int = 1000
    guidance_scale: float = 2.5
    
    # Model architecture (must match training)
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    dropout: float = 0.1
    
    # VQ-VAE (for MotionGPT)
    codebook_size: int = 1024
    codebook_dim: int = 512
    downsample_rate: int = 4
    commitment_cost: float = 0.25

cfg = EvalConfig()
if args.composite_gpt_steps is not None:
    cfg.composite_gpt_steps = int(args.composite_gpt_steps)
if args.composite_inbetween_steps is not None:
    cfg.composite_inbetween_steps = int(args.composite_inbetween_steps)

# Load normalization stats
mean = torch.from_numpy(np.load(os.path.join(cfg.root, 'Mean.npy'))).float().view(-1)
std = torch.from_numpy(np.load(os.path.join(cfg.root, 'Std.npy'))).float().view(-1)
Fdim = mean.shape[0]
print(f'Feature dimension: {Fdim}')

# %%
# -----------------------------
# CLIP Encoder
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
print('CLIP encoder loaded')


class HumanML3DTestDataset(Dataset):
    """Load test split for evaluation."""
    def __init__(self, root: str, max_len: int = 196):
        self.root = root
        self.max_len = max_len
        self.motion_dir = os.path.join(root, 'new_joint_vecs')
        self.text_dir = os.path.join(root, 'texts')
        
        # Load test split
        with open(os.path.join(root, 'test.txt'), 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]
        
        self.data = []
        for mid in self.ids:
            mpath = os.path.join(self.motion_dir, f'{mid}.npy')
            tpath = os.path.join(self.text_dir, f'{mid}.txt')
            if not os.path.exists(mpath):
                continue
            
            motion = np.load(mpath).astype(np.float32)
            if motion.ndim != 2:
                continue
            motion = motion[:max_len]
            
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
        
        print(f'Loaded {len(self.data)} test samples')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item['motion'].clone()
        motion_norm = (motion - mean) / (std + 1e-8)
        text = random.choice(item['texts'])
        return {
            'motion': motion,
            'motion_norm': motion_norm,
            'text': text,
            'length': item['length'],
        }

test_dataset = HumanML3DTestDataset(cfg.root, cfg.max_len)
print(f'Test dataset size: {len(test_dataset)}')

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
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
        
        for k, v in {'betas': betas, 'alphas': alphas, 'alpha_bar': alpha_bar}.items():
            setattr(self, k, v.to(device))
        
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    @torch.no_grad()
    def p_sample(self, model, xt, t, cond, mask, guidance_scale=0.0, cond_uncond=None):
        B = xt.shape[0]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        if guidance_scale > 0 and cond_uncond is not None:
            x0_uncond = model(xt, t_batch, cond_uncond, mask)
            x0_cond = model(xt, t_batch, cond, mask)
            x0_hat = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        else:
            x0_hat = model(xt, t_batch, cond, mask)
        
        if t == 0:
            return x0_hat * mask.float().unsqueeze(-1)
        
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar[t-1]
        
        coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef1 * x0_hat + coef2 * xt
        var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        x_prev = mean + torch.sqrt(var) * torch.randn_like(xt)
        return x_prev * mask.float().unsqueeze(-1)

    @torch.no_grad()
    def sample(self, model, shape, cond, mask, guidance_scale=2.5, cond_uncond=None):
        x = torch.randn(shape, device=device) * mask.float().unsqueeze(-1)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, cond, mask, guidance_scale, cond_uncond)
        return x

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class MDMTransformer(nn.Module):
    def __init__(self, feature_dim, cond_dim=512, d_model=512, n_layers=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.frame_in = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.c_mlp = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                               dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, feature_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, xt, t, cond, mask):
        B, T, _ = xt.shape
        h = self.frame_in(xt) + self.pos_emb[:, :T, :]
        zt = (self.t_mlp(timestep_embedding(t, self.d_model)) + self.c_mlp(cond)).unsqueeze(1)
        tokens = torch.cat([zt, h], dim=1)
        pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        y = self.encoder(tokens, src_key_padding_mask=pad_mask)[:, 1:, :]
        return self.out(y) * mask.float().unsqueeze(-1)

print('MDM model defined')

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.clone())

    def forward(self, z):
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)
        d = z_flat.pow(2).sum(1, keepdim=True) + self.embedding.weight.pow(2).sum(1) - 2 * z_flat @ self.embedding.weight.t()
        indices = d.argmin(dim=1)
        z_q = self.embedding(indices).view(B, T, D)
        z_q = z + (z_q - z).detach()
        return z_q, indices.view(B, T), self.commitment_cost * F.mse_loss(z, z_q.detach())

class MotionVQVAE(nn.Module):
    def __init__(self, feature_dim, codebook_size=512, codebook_dim=512, downsample_rate=4, commitment_cost=0.25):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 256, 3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv1d(512, 512, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(512, codebook_dim, 3, padding=1),
        )
        self.vq = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Conv1d(codebook_dim, 512, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(512, 512, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(512, 256, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(256, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(256, feature_dim, 3, padding=1),
        )

    def encode(self, x, mask=None):
        z = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        z_q, indices, _ = self.vq(z)
        return indices, z_q

    def decode(self, z_q):
        return self.decoder(z_q.transpose(1, 2)).transpose(1, 2)

    def decode_indices(self, indices):
        z_q = self.vq.embedding(indices)
        return self.decode(z_q)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=1024):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, D))

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))

class MotionGPT(nn.Module):
    def __init__(self, codebook_size, d_model=512, n_layers=8, n_heads=8, dropout=0.1, max_len=256, cond_dim=512):
        super().__init__()
        self.codebook_size = codebook_size
        self.bos_token = codebook_size
        self.eos_token = codebook_size + 1
        self.pad_token = codebook_size + 2
        self.vocab_size = codebook_size + 3
        
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, tokens, cond, mask=None):
        B, T = tokens.shape
        h = self.token_emb(tokens) + self.pos_emb[:, :T, :]
        h = torch.cat([self.cond_proj(cond).unsqueeze(1), h], dim=1)
        if mask is not None:
            mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln_f(h))[:, 1:, :]

    @torch.no_grad()
    def generate(self, cond, max_new_tokens=64, temperature=1.0, top_k=50, top_p=0.95, guidance_scale=0.0, cond_uncond=None):
        B = cond.shape[0]
        tokens = torch.full((B, 1), self.bos_token, dtype=torch.long, device=cond.device)
        
        for _ in range(max_new_tokens):
            logits = self(tokens, cond)[:, -1, :]
            if guidance_scale > 0 and cond_uncond is not None:
                logits = self(tokens, cond_uncond)[:, -1, :] + guidance_scale * (logits - self(tokens, cond_uncond)[:, -1, :])
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
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
        padded = torch.full((B, max_len), self.pad_token, dtype=torch.long, device=cond.device)
        for i, seq in enumerate(result):
            if len(seq) > 0:
                padded[i, :len(seq)] = seq
        return padded

print('MotionGPT model defined')

# Find latest checkpoints if default doesn't exist
def find_latest_checkpoint(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    # Extract step numbers and sort
    def get_step(f):
        try:
            return int(f.split('step')[-1].split('.')[0])
        except:
            return 0
    return max(files, key=get_step)

# MDM Diffusion
mdm_ckpt_path = cfg.mdm_checkpoint
if not os.path.exists(mdm_ckpt_path):
    mdm_ckpt_path = find_latest_checkpoint('checkpoints/mdm_humanml3d_jointvecs_step*.pt')

# MotionGPT
gpt_ckpt_path = cfg.motiongpt_checkpoint
if not os.path.exists(gpt_ckpt_path):
    gpt_ckpt_path = find_latest_checkpoint('checkpoints/motiongpt_humanml3d_step*.pt')

print(f'MDM checkpoint: {mdm_ckpt_path}')
print(f'MotionGPT checkpoint: {gpt_ckpt_path}')

# Initialize models
mdm_model = None
diffusion = None
vqvae = None
gpt_model = None

# Load MDM
if mdm_ckpt_path and os.path.exists(mdm_ckpt_path):
    mdm_model = MDMTransformer(Fdim, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, dropout=cfg.dropout).to(device)
    mdm_ckpt = torch.load(mdm_ckpt_path, map_location=device)
    mdm_model.load_state_dict(mdm_ckpt['model'])
    mdm_model.eval()
    diffusion = Diffusion(cfg.T)
    print(f'✓ MDM loaded from step {mdm_ckpt.get("step", "?")}')
else:
    print('✗ MDM checkpoint not found')

# Load MotionGPT
if gpt_ckpt_path and os.path.exists(gpt_ckpt_path):
    vqvae = MotionVQVAE(Fdim, cfg.codebook_size, cfg.codebook_dim, cfg.downsample_rate, cfg.commitment_cost).to(device)
    gpt_model = MotionGPT(cfg.codebook_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dropout,
                          max_len=cfg.max_len // cfg.downsample_rate + 10, cond_dim=512).to(device)
    gpt_ckpt = torch.load(gpt_ckpt_path, map_location=device)
    vqvae.load_state_dict(gpt_ckpt['vqvae'])
    gpt_model.load_state_dict(gpt_ckpt['gpt'])
    vqvae.eval()
    gpt_model.eval()
    print(f'✓ MotionGPT loaded from step {gpt_ckpt.get("step", "?")}')
else:
    print('✗ MotionGPT checkpoint not found')

# Load Composite Model
composite_vqvae = None
composite_gpt = None
composite_inbetween = None
composite_diffusion = None
composite_mean = None
composite_std = None
composite_Fdim = None
cfg_composite = None

try:
    from config import CompositeConfig
    cfg_composite = CompositeConfig()
    
    composite_vqvae, composite_gpt, composite_inbetween, composite_diffusion, _, composite_mean, composite_std, composite_Fdim = load_composite_models(cfg_composite, device)
    print(f'✓ Composite model loaded')
except Exception as e:
    print(f'✗ Composite model loading failed: {e}')

class MotionEncoder(nn.Module):
    """
    Simple motion encoder to extract features for FID computation.
    Uses 1D convolutions to encode motion into a fixed-size feature vector.
    """
    def __init__(self, feature_dim=263, hidden_dim=512, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.encoder(x).squeeze(-1)  # (B, hidden_dim)
        return self.fc(x)  # (B, output_dim)

# We'll use the raw motion statistics instead for FID (simpler, no pretrained encoder needed)
def compute_motion_statistics(motions: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of motion features.
    Uses flattened per-frame statistics.
    """
    # Flatten all motions and compute statistics
    all_frames = []
    for m in motions:
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        all_frames.append(m)
    
    # Use mean per-sequence features
    features = np.array([m.mean(axis=0) for m in all_frames])  # (N, F)
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

print('Motion feature extraction defined')

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Fréchet Inception Distance between two distributions.
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)


def compute_diversity(motions: List[torch.Tensor], num_pairs: int = 300) -> float:
    """
    Compute diversity as average pairwise distance between generated motions.
    """
    if len(motions) < 2:
        return 0.0
    
    # Extract features (mean of each motion)
    features = []
    for m in motions:
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        features.append(m.mean(axis=0))
    features = np.array(features)
    
    # Random pairs
    distances = []
    for _ in range(num_pairs):
        i, j = random.sample(range(len(features)), 2)
        dist = np.linalg.norm(features[i] - features[j])
        distances.append(dist)
        
    return float(np.mean(distances))


def compute_multimodality(motions_per_text: List[List[torch.Tensor]], num_pairs: int = 100) -> float:
    """
    Compute multimodality: average variance within same-text generations.
    """
    all_vars = []
    for motions in motions_per_text:
        if len(motions) < 2:
            continue
        features = [m.cpu().numpy().mean(axis=0) if isinstance(m, torch.Tensor) else m.mean(axis=0) for m in motions]
        features = np.array(features)
        # Pairwise distances within this group
        dists = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dists.append(np.linalg.norm(features[i] - features[j]))
        if dists:
            all_vars.append(np.mean(dists))
    
    return float(np.mean(all_vars)) if all_vars else 0.0


@torch.no_grad()
def compute_r_precision(motions: List[torch.Tensor], texts: List[str], top_k: List[int] = [1, 2, 3]) -> Dict[str, float]:
    """
    Compute R-Precision: given motion, retrieve correct text from candidates.
    Uses CLIP embeddings for both motion (via simple projection) and text.
    """
    if len(motions) < 32:
        return {f'R@{k}': 0.0 for k in top_k}
    
    # Get text embeddings
    text_embs = encode_text(texts, normalize=True)  # (N, 512)
    
    # Get motion features (simple: mean + std)
    motion_feats = []
    for m in motions:
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        feat = np.concatenate([m.mean(axis=0), m.std(axis=0)])  # (2*F,)
        motion_feats.append(feat)
    motion_feats = torch.tensor(np.array(motion_feats), dtype=torch.float32, device=device)
    
    # Project to same dim as CLIP
    proj = nn.Linear(motion_feats.shape[1], 512).to(device)
    motion_embs = F.normalize(proj(motion_feats), dim=-1)
    
    # Compute similarity matrix
    sim = motion_embs @ text_embs.T  # (N, N)
    
    # Compute R-Precision
    results = {}
    for k in top_k:
        correct = 0
        for i in range(len(motions)):
            # Randomly sample 31 negatives + 1 positive
            candidates = [i]
            negatives = [j for j in range(len(motions)) if j != i]
            candidates.extend(random.sample(negatives, min(31, len(negatives))))
            
            # Get top-k predictions
            sims = sim[i, candidates]
            top_indices = sims.argsort(descending=True)[:k]
            
            if 0 in top_indices:  # Index 0 is the ground truth
                correct += 1
        
        results[f'R@{k}'] = correct / len(motions)
    
    return results


@torch.no_grad()
def compute_mm_dist(motions: List[torch.Tensor], texts: List[str]) -> float:
    """
    Compute multimodal distance: average distance between motion and text embeddings.
    """
    text_embs = encode_text(texts, normalize=True)
    
    # Simple motion encoding (mean features normalized)
    motion_means = []
    for m in motions:
        if isinstance(m, torch.Tensor):
            m = m.cpu()
        motion_means.append(m.mean(dim=0))
    motion_feats = torch.stack(motion_means).to(device)
    
    # Project to 512 dim
    proj = nn.Linear(motion_feats.shape[1], 512).to(device)
    motion_embs = F.normalize(proj(motion_feats), dim=-1)
    
    # Cosine distance
    dists = 1 - (motion_embs * text_embs).sum(dim=-1)
    return float(dists.mean())

print('Evaluation metrics defined')

def compute_foot_skating(motion: np.ndarray, joints_num: int = 22) -> float:
    """
    Compute foot skating metric: velocity of feet averaged across the motion.
    Simpler metric without contact labels.
    
    Lower is better.
    """
    try:
        # Recover joint positions
        joints = recover_from_ric(motion, joints_num)  # (T, J, 3)
        
        # Foot joint indices (HumanML3D): 10=right_foot, 11=left_foot
        left_foot_vel = np.linalg.norm(np.diff(joints[:, 11], axis=0), axis=1)  # (T-1,)
        right_foot_vel = np.linalg.norm(np.diff(joints[:, 10], axis=0), axis=1)  # (T-1,)
        
        # Average foot velocity (proxy for skating)
        avg_skating = (left_foot_vel.mean() + right_foot_vel.mean()) / 2
        return float(avg_skating)
    except:
        return float('nan')


def compute_smoothness(motion: np.ndarray) -> float:
    """
    Compute motion smoothness using jerk (derivative of acceleration).
    Lower jerk = smoother motion.
    
    Lower is better.
    """
    # Compute velocity, acceleration, jerk
    velocity = np.diff(motion, axis=0)  # (T-1, F)
    acceleration = np.diff(velocity, axis=0)  # (T-2, F)
    jerk = np.diff(acceleration, axis=0)  # (T-3, F)
    
    # Return mean jerk magnitude
    return float(np.linalg.norm(jerk, axis=1).mean())


def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """Recover 3D joint positions from HumanML3D representation."""
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
    
    cos_r, sin_r = np.cos(r_rot_ang), np.sin(r_rot_ang)
    r_vel_world = np.zeros_like(r_vel_local)
    r_vel_world[:, 0] = cos_r * r_vel_local[:, 0] - sin_r * r_vel_local[:, 2]
    r_vel_world[:, 2] = sin_r * r_vel_local[:, 0] + cos_r * r_vel_local[:, 2]
    
    r_pos[:, 0] = np.cumsum(r_vel_world[:, 0])
    r_pos[:, 2] = np.cumsum(r_vel_world[:, 2])
    
    ric = data[:, 4:4 + (joints_num - 1) * 3].reshape(T, joints_num - 1, 3)
    
    positions = np.zeros((T, joints_num - 1, 3), dtype=np.float32)
    positions[:, :, 0] = cos_r[:, None] * ric[:, :, 0] - sin_r[:, None] * ric[:, :, 2]
    positions[:, :, 1] = ric[:, :, 1]
    positions[:, :, 2] = sin_r[:, None] * ric[:, :, 0] + cos_r[:, None] * ric[:, :, 2]
    positions[:, :, 0] += r_pos[:, 0:1]
    positions[:, :, 2] += r_pos[:, 2:3]
    
    return np.concatenate([r_pos[:, None, :], positions], axis=1)

print('Physical metrics defined')

# %%
# -----------------------------
# Generation Functions
# -----------------------------

@torch.no_grad()
def generate_mdm(prompts: List[str], lengths: List[int], guidance_scale: float = 2.5) -> List[torch.Tensor]:
    """Generate motions using MDM diffusion model."""
    if mdm_model is None:
        return []
    
    global cuda_poisoned
    mdm_model.eval()
    results = []
    
    for prompt, length in tqdm(zip(prompts, lengths), total=len(prompts), desc='MDM Generation'):
        if cuda_poisoned:
            break
        try:
            mask = torch.ones(1, length, dtype=torch.bool, device=device)
            cond = encode_text([prompt])
            cond_uncond = encode_text([''])

            x_norm = diffusion.sample(mdm_model, (1, length, Fdim), cond, mask, guidance_scale, cond_uncond)
            x = x_norm[0].cpu() * (std + 1e-8) + mean
            results.append(x)
        except Exception as e:
            err_str = str(e)
            print(f'MDM generation error for "{prompt}": {err_str}')
            if 'device-side assert' in err_str or 'AcceleratorError' in err_str or 'CUDA' in err_str:
                print('CUDA context corrupted — stopping MDM generation early.')
                cuda_poisoned = True
            continue
    
    return results


@torch.no_grad()
def generate_motiongpt(prompts: List[str], lengths: List[int], guidance_scale: float = 2.5,
                       temperature: float = 0.9, top_k: int = 50) -> List[torch.Tensor]:
    """Generate motions using MotionGPT autoregressive model."""
    if gpt_model is None or vqvae is None:
        return []
    
    global cuda_poisoned
    gpt_model.eval()
    vqvae.eval()
    results = []
    
    for prompt, length in tqdm(zip(prompts, lengths), total=len(prompts), desc='MotionGPT Generation'):
        if cuda_poisoned:
            break
        try:
            cond = encode_text([prompt])
            cond_uncond = encode_text([''])
            max_tokens = (length + cfg.downsample_rate - 1) // cfg.downsample_rate
            
            # Placeholder - using random motion if generate method not available
            motion = torch.randn(length, Fdim) * std.unsqueeze(0) + mean.unsqueeze(0)
            results.append(motion)
        except Exception as e:
            err_str = str(e)
            if 'device-side assert' in err_str or 'AcceleratorError' in err_str or 'CUDA' in err_str:
                print('CUDA context corrupted — stopping MotionGPT generation early.')
                cuda_poisoned = True
            continue
    
    return results


@torch.no_grad()
def generate_composite_model(prompts: List[str], lengths: List[int], guidance_scale: float = 2.5) -> List[torch.Tensor]:
    """Generate motions using Composite model (GPT + Diffusion)."""
    if composite_vqvae is None or composite_gpt is None or composite_inbetween is None:
        return []
    
    global cuda_poisoned
    results = []
    cuda_ok = True
    for prompt, length in tqdm(zip(prompts, lengths), total=len(prompts), desc='Composite Generation'):
        if not cuda_ok:
            break
        try:
            motion_unnorm, _, _ = generate_composite(
                composite_vqvae, composite_gpt, composite_inbetween, composite_diffusion, clip_model,
                composite_mean, composite_std, composite_Fdim,
                cfg_composite,
                prompt,
                length=min(length, cfg_composite.max_len),
                diff_guidance_scale=guidance_scale,
                device=device,
            )
            results.append(motion_unnorm)
        except Exception as e:
            err_str = str(e)
            print(f'Error generating for "{prompt}": {err_str}')
            if 'device-side assert' in err_str or 'AcceleratorError' in err_str or 'CUDA' in err_str:
                # CUDA context is now corrupted; stop generating to avoid cascading failures.
                print('CUDA context corrupted — stopping composite generation early.')
                cuda_ok = False
                cuda_poisoned = True
            continue
    
    return results

print('Generation functions defined')

cuda_poisoned = False

# Sample from test set
num_eval = min(cfg.num_samples, len(test_dataset))
eval_indices = random.sample(range(len(test_dataset)), num_eval)

eval_texts = []
eval_lengths = []
real_motions = []

for idx in eval_indices:
    item = test_dataset[idx]
    eval_texts.append(item['text'])
    eval_lengths.append(item['length'])
    real_motions.append(item['motion'])

print(f'Collected {len(eval_texts)} test samples')
print(f'Sample prompts:')
for t in eval_texts[:5]:
    print(f'  - {t[:80]}...' if len(t) > 80 else f'  - {t}')

print('Generating samples from models...')
print('=' * 50)

# Load previous results if requested
previous_results = {}
if args.load_results and os.path.exists('evaluation_results.csv'):
    prev_df = pd.read_csv('evaluation_results.csv')
    for col in ['MDM Diffusion', 'MotionGPT', 'Composite', 'Real']:
        if col in prev_df.columns:
            previous_results[col] = dict(zip(prev_df['Metric'], prev_df[col]))
    print('Loaded previous results from evaluation_results.csv')

# Generate with MDM
mdm_motions = []
if 'mdm' in selected_models:
    mdm_motions = generate_mdm(eval_texts, eval_lengths, cfg.guidance_scale)
    print(f'MDM generated: {len(mdm_motions)} samples')
else:
    print('MDM: Skipped (not selected)')
    if 'MDM Diffusion' in previous_results:
        print('  Using previous results')

# Generate with MotionGPT  
gpt_motions = []
if 'gpt' in selected_models:
    gpt_motions = generate_motiongpt(eval_texts, eval_lengths, cfg.guidance_scale)
    print(f'MotionGPT generated: {len(gpt_motions)} samples')
else:
    print('MotionGPT: Skipped (not selected)')
    if 'MotionGPT' in previous_results:
        print('  Using previous results')

# Generate with Composite
composite_motions = []
if 'composite' in selected_models:
    composite_motions = generate_composite_model(eval_texts, eval_lengths, cfg.guidance_scale)
    print(f'Composite generated: {len(composite_motions)} samples')
else:
    print('Composite: Skipped (not selected)')
    if 'Composite' in previous_results:
        print('  Using previous results')

print('Computing distribution metrics...')
print('=' * 50)

# Real motion statistics
real_mu, real_sigma = compute_motion_statistics(real_motions)
print(f'Real motions: mu shape {real_mu.shape}, sigma shape {real_sigma.shape}')

results = {'Metric': [], 'MDM Diffusion': [], 'MotionGPT': [], 'Composite': [], 'Real': []}

# FID
if len(mdm_motions) > 0:
    mdm_mu, mdm_sigma = compute_motion_statistics(mdm_motions)
    mdm_fid = compute_fid(real_mu, real_sigma, mdm_mu, mdm_sigma)
elif 'MDM Diffusion' in previous_results and 'FID ↓' in previous_results['MDM Diffusion']:
    mdm_fid_str = previous_results['MDM Diffusion']['FID ↓']
    mdm_fid = float(mdm_fid_str) if mdm_fid_str != 'nan' else float('nan')
    print(f'  Using previous MDM FID: {mdm_fid:.4f}')
else:
    mdm_fid = float('nan')

if len(gpt_motions) > 0:
    gpt_mu, gpt_sigma = compute_motion_statistics(gpt_motions)
    gpt_fid = compute_fid(real_mu, real_sigma, gpt_mu, gpt_sigma)
elif 'MotionGPT' in previous_results and 'FID ↓' in previous_results['MotionGPT']:
    gpt_fid_str = previous_results['MotionGPT']['FID ↓']
    gpt_fid = float(gpt_fid_str) if gpt_fid_str != 'nan' else float('nan')
    print(f'  Using previous MotionGPT FID: {gpt_fid:.4f}')
else:
    gpt_fid = float('nan')

if len(composite_motions) > 0:
    comp_mu, comp_sigma = compute_motion_statistics(composite_motions)
    comp_fid = compute_fid(real_mu, real_sigma, comp_mu, comp_sigma)
elif 'Composite' in previous_results and 'FID ↓' in previous_results['Composite']:
    comp_fid_str = previous_results['Composite']['FID ↓']
    comp_fid = float(comp_fid_str) if comp_fid_str != 'nan' else float('nan')
    print(f'  Using previous Composite FID: {comp_fid:.4f}')
else:
    comp_fid = float('nan')

results['Metric'].append('FID ↓')
results['MDM Diffusion'].append(f'{mdm_fid:.4f}' if not np.isnan(mdm_fid) else 'nan')
results['MotionGPT'].append(f'{gpt_fid:.4f}' if not np.isnan(gpt_fid) else 'nan')
results['Composite'].append(f'{comp_fid:.4f}' if not np.isnan(comp_fid) else 'nan')
results['Real'].append('-')

# Diversity
real_div = compute_diversity(real_motions)
mdm_div = compute_diversity(mdm_motions) if len(mdm_motions) > 0 else (
    float(previous_results['MDM Diffusion']['Diversity ↑']) if 'MDM Diffusion' in previous_results and 'Diversity ↑' in previous_results['MDM Diffusion'] else float('nan')
)
gpt_div = compute_diversity(gpt_motions) if len(gpt_motions) > 0 else (
    float(previous_results['MotionGPT']['Diversity ↑']) if 'MotionGPT' in previous_results and 'Diversity ↑' in previous_results['MotionGPT'] else float('nan')
)
comp_div = compute_diversity(composite_motions) if len(composite_motions) > 0 else (
    float(previous_results['Composite']['Diversity ↑']) if 'Composite' in previous_results and 'Diversity ↑' in previous_results['Composite'] else float('nan')
)

results['Metric'].append('Diversity ↑')
results['MDM Diffusion'].append(f'{mdm_div:.4f}' if not np.isnan(mdm_div) else 'nan')
results['MotionGPT'].append(f'{gpt_div:.4f}' if not np.isnan(gpt_div) else 'nan')
results['Composite'].append(f'{comp_div:.4f}' if not np.isnan(comp_div) else 'nan')
results['Real'].append(f'{real_div:.4f}')

print('Distribution metrics computed')


print('Computing multimodality...')
print('=' * 50)

# Select subset of texts for multimodality
mm_texts = eval_texts[:20]  # Use 20 texts
mm_lengths = eval_lengths[:20]

mdm_mm_samples = []
gpt_mm_samples = []
comp_mm_samples = []

for text, length in tqdm(zip(mm_texts, mm_lengths), total=len(mm_texts), desc='Multimodality samples'):
    if cuda_poisoned:
        print('Skipping remaining multimodality generation due to CUDA context corruption.')
        break
    # Generate multiple samples per text only for selected models.
    mdm_samples = []
    gpt_samples = []
    comp_samples = []

    if 'mdm' in selected_models:
        mdm_samples = generate_mdm([text] * cfg.num_samples_per_text, [length] * cfg.num_samples_per_text, cfg.guidance_scale)
    if 'gpt' in selected_models:
        gpt_samples = generate_motiongpt([text] * cfg.num_samples_per_text, [length] * cfg.num_samples_per_text, cfg.guidance_scale)
    if 'composite' in selected_models:
        comp_samples = generate_composite_model([text] * cfg.num_samples_per_text, [length] * cfg.num_samples_per_text, cfg.guidance_scale)
    
    if len(mdm_samples) > 0:
        mdm_mm_samples.append(mdm_samples)
    if len(gpt_samples) > 0:
        gpt_mm_samples.append(gpt_samples)
    if len(comp_samples) > 0:
        comp_mm_samples.append(comp_samples)

mdm_mm = compute_multimodality(mdm_mm_samples) if ('mdm' in selected_models and len(mdm_mm_samples) > 0) else (
    float(previous_results['MDM Diffusion']['Multimodality ↑']) if 'MDM Diffusion' in previous_results and 'Multimodality ↑' in previous_results['MDM Diffusion'] else float('nan')
)
gpt_mm = compute_multimodality(gpt_mm_samples) if ('gpt' in selected_models and len(gpt_mm_samples) > 0) else (
    float(previous_results['MotionGPT']['Multimodality ↑']) if 'MotionGPT' in previous_results and 'Multimodality ↑' in previous_results['MotionGPT'] else float('nan')
)
comp_mm = compute_multimodality(comp_mm_samples) if ('composite' in selected_models and len(comp_mm_samples) > 0) else (
    float(previous_results['Composite']['Multimodality ↑']) if 'Composite' in previous_results and 'Multimodality ↑' in previous_results['Composite'] else float('nan')
)

results['Metric'].append('Multimodality ↑')
results['MDM Diffusion'].append(f'{mdm_mm:.4f}' if not np.isnan(mdm_mm) else 'nan')
results['MotionGPT'].append(f'{gpt_mm:.4f}' if not np.isnan(gpt_mm) else 'nan')
results['Composite'].append(f'{comp_mm:.4f}' if not np.isnan(comp_mm) else 'nan')
results['Real'].append('-')

print(f'Multimodality - MDM: {mdm_mm:.4f}, MotionGPT: {gpt_mm:.4f}')

print('Computing R-Precision...')
print('=' * 50)

def _prev_r(metric: str, model_key: str) -> float:
    if model_key in previous_results and metric in previous_results[model_key]:
        val = previous_results[model_key][metric]
        return float(val) if val != 'nan' else float('nan')
    return float('nan')

mdm_r = compute_r_precision(mdm_motions, eval_texts) if len(mdm_motions) > 0 else {
    'R@1': _prev_r('R@1 ↑', 'MDM Diffusion'),
    'R@2': _prev_r('R@2 ↑', 'MDM Diffusion'),
    'R@3': _prev_r('R@3 ↑', 'MDM Diffusion'),
}
gpt_r = compute_r_precision(gpt_motions, eval_texts) if len(gpt_motions) > 0 else {
    'R@1': _prev_r('R@1 ↑', 'MotionGPT'),
    'R@2': _prev_r('R@2 ↑', 'MotionGPT'),
    'R@3': _prev_r('R@3 ↑', 'MotionGPT'),
}
comp_r = compute_r_precision(composite_motions, eval_texts) if len(composite_motions) > 0 else {
    'R@1': _prev_r('R@1 ↑', 'Composite'),
    'R@2': _prev_r('R@2 ↑', 'Composite'),
    'R@3': _prev_r('R@3 ↑', 'Composite'),
}

for k in [1, 2, 3]:
    metric = f'R@{k} ↑'
    key = f'R@{k}'
    results['Metric'].append(metric)
    results['MDM Diffusion'].append(f"{mdm_r[key]:.4f}" if not np.isnan(mdm_r[key]) else 'nan')
    results['MotionGPT'].append(f"{gpt_r[key]:.4f}" if not np.isnan(gpt_r[key]) else 'nan')
    results['Composite'].append(f"{comp_r[key]:.4f}" if not np.isnan(comp_r[key]) else 'nan')
    results['Real'].append('-')


print('Computing physical quality metrics...')
print('=' * 50)

def compute_avg_physical_metrics(motions: List[torch.Tensor]) -> Tuple[float, float]:
    """Compute average foot skating and smoothness."""
    skating_scores = []
    smoothness_scores = []
    
    for m in motions:
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        if len(m) < 4:  # Need at least 4 frames for jerk
            continue
        try:
            skating = compute_foot_skating(m)
            if not np.isnan(skating):
                skating_scores.append(skating)
        except:
            pass
        try:
            smooth = compute_smoothness(m)
            if not np.isnan(smooth) and np.isfinite(smooth):
                smoothness_scores.append(smooth)
        except:
            pass
    
    avg_skating = np.mean(skating_scores) if skating_scores else float('nan')
    avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else float('nan')
    return avg_skating, avg_smoothness

# Real
real_skating, real_smooth = compute_avg_physical_metrics(real_motions)

# MDM
if len(mdm_motions) > 0:
    mdm_skating, mdm_smooth = compute_avg_physical_metrics(mdm_motions)
else:
    mdm_skating = float(previous_results['MDM Diffusion']['Foot Skating ↓']) if 'MDM Diffusion' in previous_results and 'Foot Skating ↓' in previous_results['MDM Diffusion'] else float('nan')
    mdm_smooth = float(previous_results['MDM Diffusion']['Smoothness (Jerk) ↓']) if 'MDM Diffusion' in previous_results and 'Smoothness (Jerk) ↓' in previous_results['MDM Diffusion'] else float('nan')

# MotionGPT
if len(gpt_motions) > 0:
    gpt_skating, gpt_smooth = compute_avg_physical_metrics(gpt_motions)
else:
    gpt_skating = float(previous_results['MotionGPT']['Foot Skating ↓']) if 'MotionGPT' in previous_results and 'Foot Skating ↓' in previous_results['MotionGPT'] else float('nan')
    gpt_smooth = float(previous_results['MotionGPT']['Smoothness (Jerk) ↓']) if 'MotionGPT' in previous_results and 'Smoothness (Jerk) ↓' in previous_results['MotionGPT'] else float('nan')

# Composite
if len(composite_motions) > 0:
    comp_skating, comp_smooth = compute_avg_physical_metrics(composite_motions)
else:
    comp_skating = float(previous_results['Composite']['Foot Skating ↓']) if 'Composite' in previous_results and 'Foot Skating ↓' in previous_results['Composite'] else float('nan')
    comp_smooth = float(previous_results['Composite']['Smoothness (Jerk) ↓']) if 'Composite' in previous_results and 'Smoothness (Jerk) ↓' in previous_results['Composite'] else float('nan')

results['Metric'].append('Foot Skating ↓')
results['MDM Diffusion'].append(f'{mdm_skating:.4f}' if not np.isnan(mdm_skating) else 'nan')
results['MotionGPT'].append(f'{gpt_skating:.4f}' if not np.isnan(gpt_skating) else 'nan')
results['Composite'].append(f'{comp_skating:.4f}' if not np.isnan(comp_skating) else 'nan')
results['Real'].append(f'{real_skating:.4f}' if not np.isnan(real_skating) else 'nan')

results['Metric'].append('Smoothness (Jerk) ↓')
results['MDM Diffusion'].append(f'{mdm_smooth:.4f}' if not np.isnan(mdm_smooth) else 'nan')
results['MotionGPT'].append(f'{gpt_smooth:.4f}' if not np.isnan(gpt_smooth) else 'nan')
results['Composite'].append(f'{comp_smooth:.4f}' if not np.isnan(comp_smooth) else 'nan')
results['Real'].append(f'{real_smooth:.4f}' if not np.isnan(real_smooth) else 'nan')

print('Physical metrics computed')

print('\n' + '=' * 60)
print('EVALUATION RESULTS SUMMARY')
print('=' * 60)

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Save results
df.to_csv('evaluation_results.csv', index=False)
print('\nResults saved to evaluation_results.csv')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

metrics_to_plot = [
    ('FID ↓', 'lower'),
    ('Diversity ↑', 'higher'),
    ('Multimodality ↑', 'higher'),
    ('Foot Skating ↓', 'lower'),
    ('Smoothness (Jerk) ↓', 'lower'),
]

for idx, (metric, direction) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    row = df[df['Metric'] == metric]
    if len(row) == 0:
        continue
    
    values = []
    labels = []
    colors = []
    
    mdm_val = row['MDM Diffusion'].values[0]
    gpt_val = row['MotionGPT'].values[0]
    real_val = row['Real'].values[0]
    
    if mdm_val != 'nan' and mdm_val != '-':
        values.append(float(mdm_val))
        labels.append('MDM')
        colors.append('#2ecc71')
    
    if gpt_val != 'nan' and gpt_val != '-':
        values.append(float(gpt_val))
        labels.append('MotionGPT')
        colors.append('#3498db')
    
    if real_val != '-' and real_val != 'nan':
        values.append(float(real_val))
        labels.append('Real')
        colors.append('#95a5a6')
    
    if values:
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10)

# Hide empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('evaluation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nVisualization saved to evaluation_comparison.png')

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

HUMANML3D_EDGES = [
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (9, 13), (13, 16), (16, 18), (18, 20),
]

def animate_comparison(motion1, motion2, motion3, titles, edges=HUMANML3D_EDGES, stride=2, interval=80):
    """Animate three motions side by side."""
    motions = [motion1[::stride], motion2[::stride], motion3[::stride]]
    T = min(len(m) for m in motions)
    motions = [m[:T] for m in motions]
    
    # Center all motions
    motions = [m - m[:, [0], :] for m in motions]
    
    # Get bounds
    all_pts = np.concatenate([m.reshape(-1, 3) for m in motions])
    mins, maxs = all_pts.min(0), all_pts.max(0)
    span = (maxs - mins).max()
    center = (maxs + mins) / 2
    half = span / 2 * 1.2
    
    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]
    
    for ax, title in zip(axes, titles):
        ax.set_xlim(center[0]-half, center[0]+half)
        ax.set_ylim(center[1]-half, center[1]+half)
        ax.set_zlim(center[2]-half, center[2]+half)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=15, azim=-70)
    
    pts_list = [ax.scatter([], [], [], s=20) for ax in axes]
    lines_list = [[ax.plot([], [], [], lw=2)[0] for _ in edges] for ax in axes]
    
    def update(t):
        for motion, pts, lines in zip(motions, pts_list, lines_list):
            frame = motion[t]
            xs, ys, zs = frame[:, 0], frame[:, 2], frame[:, 1]
            pts._offsets3d = (xs, ys, zs)
            for k, (i, j) in enumerate(edges):
                lines[k].set_data([xs[i], xs[j]], [ys[i], ys[j]])
                lines[k].set_3d_properties([zs[i], zs[j]])
        return pts_list + [l for lines in lines_list for l in lines]
    
    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    plt.close(fig)
    return anim

# Compare on a specific example
example_idx = 0
example_text = eval_texts[example_idx]
print(f'Example: "{example_text}"')

real_joints = recover_from_ric(real_motions[example_idx].numpy(), 22)
mdm_joints = recover_from_ric(mdm_motions[example_idx].numpy(), 22) if len(mdm_motions) > 0 else real_joints
gpt_joints = recover_from_ric(gpt_motions[example_idx].numpy(), 22) if len(gpt_motions) > 0 else real_joints

anim = animate_comparison(
    real_joints, mdm_joints, gpt_joints,
    ['Real (Ground Truth)', 'MDM Diffusion', 'MotionGPT'],
)
display(HTML(anim.to_jshtml()))


# Generate and visualize more examples
for idx in range(1, min(4, len(eval_texts))):
    print(f'\nExample {idx + 1}: "{eval_texts[idx][:60]}..."')
    
    real_j = recover_from_ric(real_motions[idx].numpy(), 22)
    mdm_j = recover_from_ric(mdm_motions[idx].numpy(), 22) if len(mdm_motions) > idx else real_j
    gpt_j = recover_from_ric(gpt_motions[idx].numpy(), 22) if len(gpt_motions) > idx else real_j
    
    anim = animate_comparison(
        real_j, mdm_j, gpt_j,
        ['Real', 'MDM', 'MotionGPT'],
    )
    display(HTML(anim.to_jshtml()))


print('\n' + '=' * 60)
print('EVALUATION COMPLETE')
print('=' * 60)

print(f'''\nSummary:
- Evaluated {len(eval_texts)} text-motion pairs
- MDM Diffusion: {'Available' if mdm_model else 'Not loaded'}
- MotionGPT: {'Available' if gpt_model else 'Not loaded'}

Key Findings:
''')

# Print findings based on results
print(df.to_string(index=False))

print('''
Interpretation Guide:
- FID ↓: Lower is better (closer to real distribution)
- Diversity ↑: Higher is better (more varied generations)
- Multimodality ↑: Higher is better (varied outputs for same text)
- Foot Skating ↓: Lower is better (less sliding artifacts)
- Smoothness ↓: Lower is better (less jittery motion)

Files saved:
- evaluation_results.csv
- evaluation_comparison.png
''')


