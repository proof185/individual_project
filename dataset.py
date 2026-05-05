"""Dataset and data loading utilities for HUMANML3D."""

import os
import random
from collections.abc import Callable
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class HUMANML3DCompositeDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        max_len: int | None = None,
        normalize: bool = True,
        use_cache: bool = True,
        text_encoder: Callable | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        device: str = 'cuda',
    ):
        self.root = root
        self.split = split
        self.max_len = max_len
        self.normalize = normalize
        self.text_encoder = text_encoder
        self.device = device

        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            mean_path = os.path.join(root, 'Mean.npy')
            std_path = os.path.join(root, 'Std.npy')
            self.mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
            self.std = torch.from_numpy(np.load(std_path)).float().view(-1)
        self.feature_dim = int(self.mean.numel())

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
            self._validate_cached_data()
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

        # Text encoder is only needed while building cache; dropping it keeps the
        # dataset lightweight and easier to use with worker processes.
        self.text_encoder = None

    def _validate_feature_motion(self, motion, path: str, kind: str):
        shape = tuple(motion.shape)

        if len(shape) == 3 and shape[0] == 1:
            shape = shape[1:]

        if len(shape) != 2:
            raise ValueError(
                f"Expected {kind} motion in HumanML3D feature space with shape (T, {self.feature_dim}) at {path}, got {tuple(motion.shape)}"
            )
        if shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {kind} motion feature dim {self.feature_dim} at {path}, got {shape[1]}. "
                "Convert T2M-GPT XYZ outputs to HumanML3D features before training."
            )

    def _validate_cached_data(self):
        for item in self.data:
            motion = item.get('motion')
            sample_id = item.get('id', '<unknown>')
            self._validate_feature_motion(motion, f"cache:{sample_id}:motion", 'training')

    def _build_motion_cache(self):
        print('Building motion cache...')
        self.data = []
        for mid in self.ids:
            mpath = os.path.join(self.motion_dir, f'{mid}.npy')
            if not os.path.exists(mpath):
                continue
            motion = np.load(mpath).astype(np.float32)
            self._validate_feature_motion(motion, mpath, 'training')

            if self.max_len is not None:
                motion = motion[:self.max_len]

            tpath = os.path.join(self.text_dir, f'{mid}.txt')
            texts = ['']
            if os.path.exists(tpath):
                with open(tpath, 'r', encoding='utf-8') as tf:
                    texts = [ln.strip() for ln in tf if ln.strip()]
                if len(texts) == 0:
                    texts = ['']

            item = {
                'id': mid,
                'motion': torch.from_numpy(motion),
                'texts': texts,
                'length': motion.shape[0],
            }
            self.data.append(item)

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
            motion = (motion - self.mean) / (self.std + 1e-8)

        text_idx = random.randint(0, len(item['texts']) - 1)

        return {
            'id': item['id'],
            'motion': motion,
            'length': item['length'],
            'text_idx': text_idx,
        }

    def get_embedding(self, sample_id: str, text_idx: int) -> torch.Tensor:
        return self.embeddings[sample_id][text_idx]


def collate_composite(batch: List[Dict]):
    """Collate function for composite dataset."""
    motions = [b['motion'] for b in batch]
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    ids = [b['id'] for b in batch]
    text_idxs = [b['text_idx'] for b in batch]

    B = len(motions)
    T_max = int(lengths.max())
    F = motions[0].shape[1]

    x = torch.zeros(B, T_max, F, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, m in enumerate(motions):
        T = m.shape[0]
        x[i, :T] = m
        mask[i, :T] = True

    out = {
        'motion': x,
        'mask': mask,
        'lengths': lengths,
        'ids': ids,
        'text_idxs': text_idxs,
    }
    return out
