"""Dataset and data loading utilities for HUMANML3D."""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class HUMANML3DCompositeDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        max_len: Optional[int] = None,
        normalize: bool = True,
        use_cache: bool = True,
        text_encoder: Optional[callable] = None,
        keyframe_interval: int = 5,
        keyframe_strategy: str = 'interval',
        keyframe_count: Optional[int] = None,
        keyframe_min: int = 6,
        keyframe_max: int = 20,
        keyframe_include_ends: bool = True,
        include_keyframes: bool = True,
        conditioning_motion_dir: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        device: str = 'cuda',
    ):
        self.root = root
        self.split = split
        self.max_len = max_len
        self.normalize = normalize
        self.text_encoder = text_encoder
        self.keyframe_interval = keyframe_interval
        self.keyframe_strategy = keyframe_strategy
        self.keyframe_count = keyframe_count
        self.keyframe_min = keyframe_min
        self.keyframe_max = keyframe_max
        self.keyframe_include_ends = keyframe_include_ends
        self.include_keyframes = include_keyframes
        self.conditioning_motion_dir = conditioning_motion_dir
        self.device = device

        # Load mean and std if provided
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
        self.cond_motion_dir = None
        if conditioning_motion_dir:
            self.cond_motion_dir = conditioning_motion_dir
            if not os.path.isabs(self.cond_motion_dir):
                self.cond_motion_dir = os.path.join(root, self.cond_motion_dir)
            if not os.path.exists(self.cond_motion_dir):
                raise FileNotFoundError(f"Missing conditioning motion dir: {self.cond_motion_dir}")
        self.text_dir = os.path.join(root, 'texts')
        self.split_file = os.path.join(root, f'{split}.txt')

        if not os.path.exists(self.motion_dir):
            raise FileNotFoundError(f"Missing {self.motion_dir}")

        with open(self.split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

        cond_tag = 'none' if self.cond_motion_dir is None else os.path.basename(self.cond_motion_dir.rstrip('/\\'))
        motion_cache = f"HUMANML3D_joint_vecs_{split}_maxlen{max_len}_norm{normalize}_cond{cond_tag}.pt"
        self.motion_cache_path = os.path.join(root, motion_cache)
        self.emb_cache_path = os.path.join(root, f'clip_embeddings_{split}.pt')
        self.token_emb_cache_path = os.path.join(root, f'clip_token_embeddings_{split}.pt')

        self.data: List[Dict] = []
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.token_embeddings: Dict[str, torch.Tensor] = {}
        self.token_masks: Dict[str, torch.Tensor] = {}

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

        if text_encoder is not None and hasattr(text_encoder, 'encode_with_tokens'):
            if use_cache and os.path.exists(self.token_emb_cache_path):
                print(f'Loading cached token-level CLIP embeddings from {self.token_emb_cache_path}')
                cpu_cache = torch.load(self.token_emb_cache_path)
                for mid, payload in cpu_cache.items():
                    self.token_embeddings[mid] = payload['features'].to(device)
                    self.token_masks[mid] = payload['mask'].to(device)
                print(f'Loaded {len(self.token_embeddings)} token feature entries')
            else:
                self._build_token_embedding_cache()
                if use_cache:
                    cpu_cache = {
                        mid: {
                            'features': self.token_embeddings[mid].cpu(),
                            'mask': self.token_masks[mid].cpu(),
                        }
                        for mid in self.token_embeddings.keys()
                    }
                    torch.save(cpu_cache, self.token_emb_cache_path)

        # Text encoder is only needed while building cache; dropping it keeps the
        # dataset lightweight and easier to use with worker processes.
        self.text_encoder = None

    def _validate_feature_motion(self, motion, path: str, kind: str):
        if isinstance(motion, torch.Tensor):
            shape = tuple(motion.shape)
        else:
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
            if 'conditioning_motion' in item:
                self._validate_feature_motion(
                    item['conditioning_motion'],
                    f"cache:{sample_id}:conditioning",
                    'conditioning',
                )

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

            conditioning_motion = None
            if self.cond_motion_dir is not None:
                cpath = os.path.join(self.cond_motion_dir, f'{mid}.npy')
                if not os.path.exists(cpath):
                    continue
                conditioning_motion = np.load(cpath).astype(np.float32)
                if conditioning_motion.ndim == 3 and conditioning_motion.shape[0] == 1:
                    conditioning_motion = conditioning_motion[0]
                self._validate_feature_motion(conditioning_motion, cpath, 'conditioning')
                if self.max_len is not None:
                    conditioning_motion = conditioning_motion[:self.max_len]

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
            if conditioning_motion is not None:
                item['conditioning_motion'] = torch.from_numpy(conditioning_motion)
            self.data.append(item)

    @torch.no_grad()
    def _build_embedding_cache(self):
        print('Pre-computing CLIP embeddings...')
        for item in self.data:
            mid = item['id']
            texts = item['texts']
            embs = self.text_encoder(texts, normalize=True)
            self.embeddings[mid] = embs

    @torch.no_grad()
    def _build_token_embedding_cache(self):
        print('Pre-computing token-level CLIP embeddings...')
        for item in self.data:
            mid = item['id']
            texts = item['texts']
            _, token_features, token_mask = self.text_encoder.encode_with_tokens(texts, normalize=True)
            self.token_embeddings[mid] = token_features
            self.token_masks[mid] = token_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item['motion'].clone()
        if self.cond_motion_dir is not None:
            conditioning_motion = item['conditioning_motion'].clone()
        else:
            conditioning_motion = item['motion'].clone()

        if self.normalize:
            motion = (motion - self.mean) / (self.std + 1e-8)
            conditioning_motion = (conditioning_motion - self.mean) / (self.std + 1e-8)

        if conditioning_motion.shape[0] != motion.shape[0]:
            conditioning_motion = self._resample_motion_to_length(conditioning_motion, motion.shape[0])

        text_idx = random.randint(0, len(item['texts']) - 1)

        out = {
            'id': item['id'],
            'motion': motion,
            'length': item['length'],
            'text_idx': text_idx,
        }

        if self.include_keyframes:
            T = motion.shape[0]
            if self.keyframe_strategy == 'random':
                keyframe_indices = self._sample_random_keyframes(T)
            else:
                # Extract keyframes (every n frames)
                keyframe_indices = list(range(0, T, self.keyframe_interval))
                # Always include the last frame
                if keyframe_indices[-1] != T - 1:
                    keyframe_indices.append(T - 1)
            keyframe_indices = torch.tensor(keyframe_indices, dtype=torch.long)
            keyframes = conditioning_motion[keyframe_indices]  # (K, F)
            out['keyframes'] = keyframes
            out['keyframe_indices'] = keyframe_indices

        return out

    def get_embedding(self, sample_id: str, text_idx: int) -> torch.Tensor:
        return self.embeddings[sample_id][text_idx]

    def get_token_embedding(self, sample_id: str, text_idx: int) -> torch.Tensor:
        return self.token_embeddings[sample_id][text_idx]

    def get_token_mask(self, sample_id: str, text_idx: int) -> torch.Tensor:
        return self.token_masks[sample_id][text_idx]

    def _sample_random_keyframes(self, length: int) -> List[int]:
        if length <= 0:
            return [0]

        if self.keyframe_count is not None:
            k = self.keyframe_count
        else:
            k = random.randint(self.keyframe_min, self.keyframe_max)

        if self.keyframe_include_ends:
            k = max(k, 2)

        k = max(1, min(k, length))

        if self.keyframe_include_ends and length >= 2:
            indices = {0, length - 1}
            remaining = [i for i in range(1, length - 1)]
            k_remaining = max(0, k - len(indices))
            if k_remaining > 0 and len(remaining) > 0:
                indices.update(random.sample(remaining, min(k_remaining, len(remaining))))
        else:
            indices = set(random.sample(range(length), k))

        return sorted(indices)

    @staticmethod
    def _resample_motion_to_length(motion: torch.Tensor, target_len: int) -> torch.Tensor:
        if motion.shape[0] == target_len:
            return motion
        if target_len <= 1:
            return motion[:1]
        src = motion.cpu().numpy()
        old_t, feat_dim = src.shape
        if old_t <= 1:
            return torch.from_numpy(np.repeat(src[:1], target_len, axis=0)).float()
        x_old = np.linspace(0.0, 1.0, old_t, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        out = np.stack([np.interp(x_new, x_old, src[:, j]) for j in range(feat_dim)], axis=1)
        return torch.from_numpy(out.astype(np.float32))


def collate_composite(batch: List[Dict]):
    """Collate function for composite dataset."""
    motions = [b['motion'] for b in batch]
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    ids = [b['id'] for b in batch]
    text_idxs = [b['text_idx'] for b in batch]
    has_keyframes = 'keyframes' in batch[0] and 'keyframe_indices' in batch[0]
    keyframes_list = [b['keyframes'] for b in batch] if has_keyframes else None
    keyframe_indices_list = [b['keyframe_indices'] for b in batch] if has_keyframes else None

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

    if has_keyframes:
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

        out['keyframes'] = keyframes
        out['keyframe_mask'] = keyframe_mask
        out['keyframe_indices'] = keyframe_indices

    return out
