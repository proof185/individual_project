"""Compute only R-Precision for pre-generated motions."""

import os
import argparse
import random
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


def load_motions(path: str) -> List[torch.Tensor]:
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
        return [torch.from_numpy(np.load(os.path.join(path, f))) for f in files]
    if path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        motions = list(data['motions'])
        return [torch.from_numpy(m) if isinstance(m, np.ndarray) else m for m in motions]
    if path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            return [torch.from_numpy(m) for m in arr]
        return [torch.from_numpy(arr)]
    raise ValueError('Unsupported motions format. Use .npz, .npy, or a folder of .npy files.')


def load_texts(root: str, split: str, num_samples: int, seed: int) -> List[str]:
    split_path = os.path.join(root, f'{split}.txt')
    text_dir = os.path.join(root, 'texts')

    with open(split_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]

    random.seed(seed)
    indices = random.sample(range(len(ids)), min(num_samples, len(ids)))

    texts = []
    for idx in indices:
        mid = ids[idx]
        tpath = os.path.join(text_dir, f'{mid}.txt')
        candidates = ['']
        if os.path.exists(tpath):
            with open(tpath, 'r', encoding='utf-8') as tf:
                candidates = [ln.strip() for ln in tf if ln.strip()]
            if len(candidates) == 0:
                candidates = ['']
        texts.append(random.choice(candidates))

    return texts


def compute_r_precision(motions: List[torch.Tensor], texts: List[str], device: str, top_k: List[int] = [1, 2, 3]) -> Dict[str, float]:
    if len(motions) < 32:
        return {f'R@{k}': 0.0 for k in top_k}

    clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    tokens = clip.tokenize(texts, truncate=True).to(device)
    text_embs = clip_model.encode_text(tokens).float()
    text_embs = F.normalize(text_embs, dim=-1)

    motion_feats = []
    for m in motions:
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        feat = np.concatenate([m.mean(axis=0), m.std(axis=0)])
        motion_feats.append(feat)
    motion_feats = torch.tensor(np.array(motion_feats), dtype=torch.float32, device=device)

    proj = nn.Linear(motion_feats.shape[1], 512).to(device)
    motion_embs = F.normalize(proj(motion_feats), dim=-1)

    sim = motion_embs @ text_embs.T

    results = {}
    for k in top_k:
        correct = 0
        for i in range(len(motions)):
            candidates = [i]
            negatives = [j for j in range(len(motions)) if j != i]
            candidates.extend(random.sample(negatives, min(31, len(negatives))))

            sims = sim[i, candidates]
            top_indices = sims.argsort(descending=True)[:k]
            if 0 in top_indices:
                correct += 1
        results[f'R@{k}'] = correct / len(motions)

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute R-Precision only')
    parser.add_argument('--motions', required=True, help='Path to motions (.npz/.npy) or folder of .npy files')
    parser.add_argument('--root', default='humanml', help='Dataset root')
    parser.add_argument('--split', default='test', help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=256, help='Number of samples')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for text selection')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    motions = load_motions(args.motions)
    texts = load_texts(args.root, args.split, args.num_samples, args.seed)

    if len(motions) != len(texts):
        print(f'Warning: motions count ({len(motions)}) != texts count ({len(texts)})')

    r = compute_r_precision(motions[:len(texts)], texts, device)
    print(f"R-Precision: R@1={r['R@1']:.4f} | R@2={r['R@2']:.4f} | R@3={r['R@3']:.4f}")


if __name__ == '__main__':
    main()
