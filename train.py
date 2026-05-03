"""Training script for composite motion generation."""

import argparse
import csv
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from datetime import datetime
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from config import CompositeConfig
from dataset import HUMANML3DCompositeDataset, collate_composite
from models.vqvae import MotionVQVAE
from models.gpt import MotionGPT
from models.diffusion import InbetweenDiffusion, InbetweenTransformer
from models.selectors import SELECTOR_MODE_CHOICES, build_keyframe_selector
from utils import (
    boundary_jerk_loss,
    encode_text,
    encode_text_with_tokens,
    masked_mse,
    prepare_gpt_batch,
    setup_clip_model,
    vel_loss,
    weighted_masked_mse,
)


class TextEncoderWrapper:
    """Wrapper for text encoder that can be pickled for multiprocessing."""
    def __init__(self, clip_model):
        self.clip_model = clip_model
    
    def __call__(self, texts, normalize=True):
        return encode_text(self.clip_model, texts, normalize)

    def encode_with_tokens(self, texts, normalize=True):
        return encode_text_with_tokens(self.clip_model, texts, normalize)


def _count_prefix_layers(state_dict: dict, prefix: str, index_pos: int) -> int:
    idx = set()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        parts = key.split('.')
        if len(parts) > index_pos and parts[index_pos].isdigit():
            idx.add(int(parts[index_pos]))
    return (max(idx) + 1) if idx else 0


def _apply_inbetween_arch_from_checkpoint(cfg, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get('cfg') if isinstance(checkpoint, dict) else None
    state = checkpoint.get('inbetween') if isinstance(checkpoint, dict) else None
    selector_state = checkpoint.get('selector') if isinstance(checkpoint, dict) else None

    if isinstance(saved_cfg, dict):
        cfg.inbetween_d_model = int(saved_cfg.get('inbetween_d_model', saved_cfg.get('d_model', cfg.inbetween_d_model)))
        cfg.inbetween_layers = int(saved_cfg.get('inbetween_layers', saved_cfg.get('n_layers', cfg.inbetween_layers)))
        cfg.inbetween_heads = int(saved_cfg.get('inbetween_heads', saved_cfg.get('n_heads', cfg.inbetween_heads)))
        saved_selector_mode = saved_cfg.get('selector_mode')
        if saved_selector_mode:
            cfg.selector_mode = str(saved_selector_mode)
        cfg.selector_d_model = int(saved_cfg.get('selector_d_model', cfg.selector_d_model))
        cfg.selector_layers = int(saved_cfg.get('selector_layers', cfg.selector_layers))
        cfg.selector_heads = int(saved_cfg.get('selector_heads', cfg.selector_heads))
        cfg.selector_aux_weight = float(saved_cfg.get('selector_aux_weight', cfg.selector_aux_weight))

    if isinstance(state, dict):
        if 'frame_in.weight' in state:
            cfg.inbetween_d_model = int(state['frame_in.weight'].shape[0])
        counted = _count_prefix_layers(state, 'encoder.layers.', 2)
        if counted > 0:
            cfg.inbetween_layers = counted

    if isinstance(selector_state, dict):
        if 'frame_in.weight' in selector_state:
            cfg.selector_d_model = int(selector_state['frame_in.weight'].shape[0])
        counted = _count_prefix_layers(selector_state, 'encoder.layers.', 2)
        if counted > 0:
            cfg.selector_layers = counted


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().item()
    try:
        return float(value)
    except Exception:
        return None


def _default_selector_oracle_ckpt_path() -> str | None:
    preferred = [
        os.path.join('checkpoints', 'composite_inbetween_text_alignment_best.pt'),
        os.path.join('checkpoints', 'composite_inbetween_transformer_best.pt'),
        os.path.join('checkpoints', 'composite_inbetween_best.pt'),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    candidates = []
    ckpt_dir = 'checkpoints'
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            if name.startswith('composite_inbetween_text_alignment_step') and name.endswith('.pt'):
                candidates.append(os.path.join(ckpt_dir, name))
            if name.startswith('composite_inbetween_transformer_step') and name.endswith('.pt'):
                candidates.append(os.path.join(ckpt_dir, name))
    if not candidates:
        return None

    def _extract_step(path: str) -> int:
        base = os.path.basename(path)
        step = base
        for prefix in ('composite_inbetween_text_alignment_step', 'composite_inbetween_transformer_step'):
            if step.startswith(prefix):
                step = step.removeprefix(prefix)
                break
        step = step.removesuffix('.pt')
        return int(step) if step.isdigit() else -1

    candidates.sort(key=_extract_step)
    return candidates[-1]


def _default_selector_eval_root() -> str | None:
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'T2M-GPT')),
        os.path.abspath(os.path.join('..', 'T2M-GPT')),
        os.path.abspath(os.path.join('T2M-GPT')),
        os.path.abspath('D:/Projects/T2M-GPT'),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _selector_oracle_cache_path(cfg, split: str, oracle_ckpt_path: str) -> str:
    os.makedirs(os.path.join(cfg.root, 'oracle_labels'), exist_ok=True)
    ckpt_tag = os.path.splitext(os.path.basename(oracle_ckpt_path))[0]
    ratio_tag = int(round(float(cfg.selector_target_ratio) * 1000))
    timestep_tag = int(getattr(cfg, 'selector_oracle_timesteps', 3))
    selector_tag = str(getattr(cfg, 'selector_mode', 'selector')).strip().lower().replace('-', '_')
    filename = f'{selector_tag}_{split}_{ckpt_tag}_r{ratio_tag}_t{timestep_tag}.pt'
    return os.path.join(cfg.root, 'oracle_labels', filename)


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def _temporary_t2mgpt_imports(t2mgpt_root: str):
    package_paths = {
        'dataset': os.path.join(t2mgpt_root, 'dataset'),
        'models': os.path.join(t2mgpt_root, 'models'),
        'options': os.path.join(t2mgpt_root, 'options'),
        'utils': os.path.join(t2mgpt_root, 'utils'),
    }
    old_modules = {name: sys.modules.get(name) for name in package_paths}
    inserted_root = False
    if t2mgpt_root not in sys.path:
        sys.path.insert(0, t2mgpt_root)
        inserted_root = True
    try:
        for name, path in package_paths.items():
            module = types.ModuleType(name)
            module.__path__ = [path]  # type: ignore[attr-defined]
            sys.modules[name] = module
        yield
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
        if inserted_root:
            sys.path.remove(t2mgpt_root)


def _load_retrieval_eval_components(t2mgpt_root: str, device: torch.device):
    with _temporary_t2mgpt_imports(t2mgpt_root), _pushd(t2mgpt_root):
        from models.evaluator_wrapper import EvaluatorModelWrapper  # type: ignore
        from options.get_eval_option import get_opt  # type: ignore
        from utils.word_vectorizer import WordVectorizer  # type: ignore

        dataset_opt_path = os.path.join(
            t2mgpt_root,
            'checkpoints',
            't2m',
            'Comp_v6_KLD005',
            'opt.txt',
        )
        wrapper_opt = get_opt(dataset_opt_path, device)
        wrapper_opt.checkpoints_dir = os.path.join(t2mgpt_root, 'checkpoints')
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        word_vectorizer = WordVectorizer('./glove', 'our_vab')
    return eval_wrapper, word_vectorizer


def _load_oracle_inbetween_model(
    oracle_ckpt_path: str,
    feature_dim: int,
    device: torch.device,
):
    oracle_cfg = CompositeConfig()
    _apply_inbetween_arch_from_checkpoint(oracle_cfg, oracle_ckpt_path, device)
    oracle_model = InbetweenTransformer(
        feature_dim=feature_dim,
        cond_dim=512,
        d_model=oracle_cfg.inbetween_d_model,
        n_layers=oracle_cfg.inbetween_layers,
        n_heads=oracle_cfg.inbetween_heads,
        dropout=oracle_cfg.dropout,
        max_len=oracle_cfg.max_len + 10,
    ).to(device)
    oracle_diff = InbetweenDiffusion(oracle_cfg.T_diffusion, device=device)

    checkpoint = torch.load(oracle_ckpt_path, map_location=device)
    state = checkpoint.get('inbetween_ema', checkpoint.get('inbetween'))
    if state is None:
        raise ValueError(f'Missing inbetween weights in oracle checkpoint: {oracle_ckpt_path}')
    oracle_model.load_state_dict(state)
    oracle_model.eval()
    return oracle_model, oracle_diff


def _parse_selector_text_entry(text_entry: str) -> tuple[str, list[str]]:
    parts = [part.strip() for part in text_entry.split('#')]
    caption = parts[0] if parts else ''
    if len(parts) > 1 and parts[1]:
        tokens = [tok for tok in parts[1].split(' ') if tok]
    else:
        tokens = [f'{word.lower()}/OTHER' for word in caption.split() if word.strip()]
    return caption, tokens


def _vectorize_retrieval_tokens(tokens: list[str], word_vectorizer, max_text_len: int = 20):
    if len(tokens) < max_text_len:
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)

    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        word_emb, pos_oh = word_vectorizer[token]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])

    pos_tensor = torch.from_numpy(np.concatenate(pos_one_hots, axis=0)).float()
    word_tensor = torch.from_numpy(np.concatenate(word_embeddings, axis=0)).float()
    return word_tensor, pos_tensor, sent_len


def _evaluator_text_embeddings(eval_wrapper, word_embs: torch.Tensor, pos_ohot: torch.Tensor, cap_lens: torch.Tensor):
    with torch.no_grad():
        return eval_wrapper.text_encoder(
            word_embs.to(eval_wrapper.device).float(),
            pos_ohot.to(eval_wrapper.device).float(),
            cap_lens.to(eval_wrapper.device),
        )


def _evaluator_motion_embeddings(eval_wrapper, motions: torch.Tensor, m_lens: torch.Tensor):
    with torch.no_grad():
        motions = motions.to(eval_wrapper.device).float()
        m_lens = m_lens.to(eval_wrapper.device).long()
        movements = eval_wrapper.movement_encoder(motions[..., :-4]).detach()
        enc_lens = (m_lens // eval_wrapper.opt.unit_length).clamp(min=1)
        return eval_wrapper.motion_encoder(movements, enc_lens)


def _build_retrieval_text_embedding_cache(dataset, eval_wrapper, word_vectorizer) -> dict[str, torch.Tensor]:
    cache = {}
    for item in dataset.data:
        embeddings = []
        for text_entry in item['texts']:
            _, tokens = _parse_selector_text_entry(text_entry)
            word_embs, pos_ohot, sent_len = _vectorize_retrieval_tokens(tokens, word_vectorizer)
            text_embedding = _evaluator_text_embeddings(
                eval_wrapper,
                word_embs.unsqueeze(0),
                pos_ohot.unsqueeze(0),
                torch.tensor([sent_len], dtype=torch.long),
            )
            embeddings.append(text_embedding.squeeze(0).detach().cpu())
        cache[item['id']] = torch.stack(embeddings, dim=0)
    return cache


def _sample_retrieval_negative_embeddings(
    sample_id: str,
    text_embedding_cache: dict[str, torch.Tensor],
    max_negatives: int,
) -> torch.Tensor:
    candidate_ids = [other_id for other_id in text_embedding_cache.keys() if other_id != sample_id]
    if not candidate_ids:
        return torch.empty(0, 512)
    rng = random.Random(f'retrieval_gain:{sample_id}:{max_negatives}')
    chosen_ids = rng.sample(candidate_ids, k=min(max_negatives, len(candidate_ids)))
    negatives = [text_embedding_cache[other_id].mean(dim=0) for other_id in chosen_ids]
    return torch.stack(negatives, dim=0)


def _retrieval_margin_from_motion_embeddings(
    positive_text_embedding: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    motion_embeddings: torch.Tensor,
) -> torch.Tensor:
    pos_dist = torch.norm(motion_embeddings - positive_text_embedding.unsqueeze(0), dim=-1)
    if negative_text_embeddings.numel() == 0:
        return -pos_dist
    neg_dist = torch.cdist(motion_embeddings, negative_text_embeddings, p=2.0)
    best_negative = neg_dist.min(dim=1).values
    return best_negative - pos_dist


def _selector_budget_from_length(length: int, ratio: float) -> int:
    if length <= 0:
        return 1
    min_budget = 3 if length > 2 else length
    return max(min_budget, min(length, int(round(length * float(ratio)))))


def _temporal_diff_norm(x: torch.Tensor, order: int) -> torch.Tensor:
    out = x
    for _ in range(order):
        out = out[:, 1:, :] - out[:, :-1, :]
    if out.shape[1] == 0:
        return x.new_zeros(x.shape[0], x.shape[1])
    score = out.norm(dim=-1)
    if order > 0:
        score = torch.nn.functional.pad(score, (order, 0))
    return score[:, : x.shape[1]]


def _information_gain_frame_weights(x0: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    velocity = _temporal_diff_norm(x0, order=1) * valid_mask.float()
    acceleration = _temporal_diff_norm(x0, order=2) * valid_mask.float()
    score = velocity + 0.5 * acceleration
    score = score / score.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return (1.0 + score) * valid_mask.float()


def _sparse_keyframes_from_indices(x0: torch.Tensor, indices: list[int], device: torch.device):
    idx = torch.tensor(sorted(set(indices)), dtype=torch.long, device=device)
    keyframes = x0.index_select(0, idx).unsqueeze(0)
    keyframe_indices = idx.unsqueeze(0)
    keyframe_mask = torch.ones(1, idx.numel(), dtype=torch.bool, device=device)
    return keyframes, keyframe_indices, keyframe_mask


def _oracle_loss_for_observed_indices(
    oracle_model,
    oracle_diff,
    x0: torch.Tensor,
    valid_mask: torch.Tensor,
    cond: torch.Tensor,
    observed_indices: list[int],
    frame_weights: torch.Tensor,
    oracle_timesteps: list[int],
) -> float:
    x0_batch = x0.unsqueeze(0)
    mask_batch = valid_mask.unsqueeze(0)
    keyframes, keyframe_indices, keyframe_mask = _sparse_keyframes_from_indices(x0, observed_indices, x0.device)

    losses = []
    for t in oracle_timesteps:
        t_batch = torch.full((1,), int(t), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0_batch)
        xt = oracle_diff.q_sample(x0_batch, t_batch, noise)
        xt = oracle_diff._replace_keyframes(xt, keyframes, keyframe_indices, keyframe_mask)
        x0_hat = oracle_model(xt, t_batch, cond, mask_batch, keyframes, keyframe_indices, keyframe_mask)

        obs_mask = torch.zeros_like(valid_mask, dtype=torch.float32)
        obs_mask[keyframe_indices[0][keyframe_mask[0]]] = 1.0
        weights = frame_weights * (1.0 - obs_mask)
        losses.append(weighted_masked_mse(x0_batch, x0_hat, weights.unsqueeze(0), mask_batch).item())

    return float(sum(losses) / max(1, len(losses)))


def _oracle_candidate_losses(
    oracle_model,
    oracle_diff,
    x0: torch.Tensor,
    valid_mask: torch.Tensor,
    cond: torch.Tensor,
    observed_indices: list[int],
    candidate_indices: list[int],
    frame_weights: torch.Tensor,
    oracle_timesteps: list[int],
    chunk_size: int = 64,
) -> torch.Tensor:
    if not candidate_indices:
        return torch.empty(0, device=x0.device)

    losses_out = []
    T = x0.shape[0]
    mask_batch_template = valid_mask.unsqueeze(0)
    x0_single = x0.unsqueeze(0)

    for start in range(0, len(candidate_indices), chunk_size):
        chunk = candidate_indices[start:start + chunk_size]
        C = len(chunk)
        mask_batch = mask_batch_template.expand(C, -1)
        x0_batch = x0_single.expand(C, -1, -1)
        obs_mask = torch.zeros(C, T, device=x0.device, dtype=torch.float32)
        if observed_indices:
            obs_mask[:, observed_indices] = 1.0
        obs_mask[torch.arange(C, device=x0.device), torch.tensor(chunk, device=x0.device)] = 1.0

        keyframe_canvas = x0_batch
        weights = frame_weights.unsqueeze(0).expand(C, -1) * (1.0 - obs_mask)
        loss_accum = 0.0

        for t in oracle_timesteps:
            t_batch = torch.full((C,), int(t), device=x0.device, dtype=torch.long)
            noise = torch.randn_like(x0_batch)
            xt = oracle_diff.q_sample(x0_batch, t_batch, noise)
            xt = oracle_diff._replace_keyframes(
                xt,
                observation_mask=obs_mask,
                keyframe_canvas=keyframe_canvas,
            )
            x0_hat = oracle_model(
                xt,
                t_batch,
                cond.expand(C, -1),
                mask_batch,
                observation_mask=obs_mask,
                keyframe_canvas=keyframe_canvas,
            )
            loss_accum = loss_accum + ((x0_batch - x0_hat) ** 2 * weights.unsqueeze(-1)).sum(dim=(1, 2)) / (
                weights.sum(dim=1).clamp(min=1e-8) * x0.shape[-1]
            )

        losses_out.append(loss_accum / max(1, len(oracle_timesteps)))

    return torch.cat(losses_out, dim=0)


def _oracle_retrieval_margin_for_observed_indices(
    oracle_model,
    oracle_diff,
    eval_wrapper,
    x0: torch.Tensor,
    valid_mask: torch.Tensor,
    cond: torch.Tensor,
    observed_indices: list[int],
    positive_text_embedding: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    oracle_timesteps: list[int],
) -> float:
    x0_batch = x0.unsqueeze(0)
    mask_batch = valid_mask.unsqueeze(0)
    obs_mask = torch.zeros_like(valid_mask, dtype=torch.float32)
    obs_mask[observed_indices] = 1.0
    obs_mask_batch = obs_mask.unsqueeze(0)
    keyframe_canvas = x0_batch
    motion_lengths = torch.tensor([x0.shape[0]], dtype=torch.long, device=x0.device)

    margins = []
    for t in oracle_timesteps:
        t_batch = torch.full((1,), int(t), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0_batch)
        xt = oracle_diff.q_sample(x0_batch, t_batch, noise)
        xt = oracle_diff._replace_keyframes(
            xt,
            observation_mask=obs_mask_batch,
            keyframe_canvas=keyframe_canvas,
        )
        x0_hat = oracle_model(
            xt,
            t_batch,
            cond,
            mask_batch,
            observation_mask=obs_mask_batch,
            keyframe_canvas=keyframe_canvas,
        )
        motion_embeddings = _evaluator_motion_embeddings(eval_wrapper, x0_hat, motion_lengths)
        margin = _retrieval_margin_from_motion_embeddings(
            positive_text_embedding,
            negative_text_embeddings,
            motion_embeddings,
        )
        margins.append(float(margin[0].item()))
    return float(sum(margins) / max(1, len(margins)))


def _oracle_candidate_retrieval_margins(
    oracle_model,
    oracle_diff,
    eval_wrapper,
    x0: torch.Tensor,
    valid_mask: torch.Tensor,
    cond: torch.Tensor,
    observed_indices: list[int],
    candidate_indices: list[int],
    positive_text_embedding: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    oracle_timesteps: list[int],
    chunk_size: int = 64,
) -> torch.Tensor:
    if not candidate_indices:
        return torch.empty(0, device=x0.device)

    margins_out = []
    T = x0.shape[0]
    mask_batch_template = valid_mask.unsqueeze(0)
    x0_single = x0.unsqueeze(0)

    for start in range(0, len(candidate_indices), chunk_size):
        chunk = candidate_indices[start:start + chunk_size]
        C = len(chunk)
        mask_batch = mask_batch_template.expand(C, -1)
        x0_batch = x0_single.expand(C, -1, -1)
        obs_mask = torch.zeros(C, T, device=x0.device, dtype=torch.float32)
        if observed_indices:
            obs_mask[:, observed_indices] = 1.0
        obs_mask[torch.arange(C, device=x0.device), torch.tensor(chunk, device=x0.device)] = 1.0

        keyframe_canvas = x0_batch
        motion_lengths = torch.full((C,), x0.shape[0], dtype=torch.long, device=x0.device)
        margin_accum = 0.0

        for t in oracle_timesteps:
            t_batch = torch.full((C,), int(t), device=x0.device, dtype=torch.long)
            noise = torch.randn_like(x0_batch)
            xt = oracle_diff.q_sample(x0_batch, t_batch, noise)
            xt = oracle_diff._replace_keyframes(
                xt,
                observation_mask=obs_mask,
                keyframe_canvas=keyframe_canvas,
            )
            x0_hat = oracle_model(
                xt,
                t_batch,
                cond.expand(C, -1),
                mask_batch,
                observation_mask=obs_mask,
                keyframe_canvas=keyframe_canvas,
            )
            motion_embeddings = _evaluator_motion_embeddings(eval_wrapper, x0_hat, motion_lengths)
            margin_accum = margin_accum + _retrieval_margin_from_motion_embeddings(
                positive_text_embedding,
                negative_text_embeddings,
                motion_embeddings,
            )

        margins_out.append(margin_accum / max(1, len(oracle_timesteps)))

    return torch.cat(margins_out, dim=0)


def _build_information_gain_target_for_item(
    oracle_model,
    oracle_diff,
    motion: torch.Tensor,
    cond: torch.Tensor,
    selector_target_ratio: float,
    oracle_timesteps: list[int],
) -> torch.Tensor:
    x0 = motion.float()
    valid_mask = torch.ones(x0.shape[0], dtype=torch.bool, device=x0.device)
    budget = _selector_budget_from_length(int(valid_mask.sum().item()), selector_target_ratio)
    observed = [0, max(0, x0.shape[0] - 1)]
    target = torch.zeros(x0.shape[0], dtype=torch.float32, device=x0.device)
    target[observed] = 1.0
    frame_weights = _information_gain_frame_weights(x0.unsqueeze(0), valid_mask.unsqueeze(0))[0]

    baseline_loss = _oracle_loss_for_observed_indices(
        oracle_model,
        oracle_diff,
        x0,
        valid_mask,
        cond,
        observed,
        frame_weights,
        oracle_timesteps,
    )

    while len(observed) < budget:
        candidates = [idx for idx in range(1, x0.shape[0] - 1) if idx not in observed]
        if not candidates:
            break
        candidate_losses = _oracle_candidate_losses(
            oracle_model,
            oracle_diff,
            x0,
            valid_mask,
            cond,
            observed,
            candidates,
            frame_weights,
            oracle_timesteps,
        )
        gains = baseline_loss - candidate_losses
        best_pos = int(torch.argmax(gains).item())
        best_idx = candidates[best_pos]
        best_gain = float(max(gains[best_pos].item(), 0.0))
        observed.append(best_idx)
        baseline_loss = float(candidate_losses[best_pos].item())
        target[best_idx] = max(target[best_idx].item(), best_gain)

    target = target * valid_mask.float()
    max_val = target.max().clamp(min=1e-6)
    target = target / max_val
    target[0] = 1.0
    target[max(0, x0.shape[0] - 1)] = 1.0
    return target.cpu()


def _build_retrieval_gain_target_for_item(
    oracle_model,
    oracle_diff,
    eval_wrapper,
    motion: torch.Tensor,
    cond: torch.Tensor,
    positive_text_embedding: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    selector_target_ratio: float,
    oracle_timesteps: list[int],
) -> torch.Tensor:
    x0 = motion.float()
    valid_mask = torch.ones(x0.shape[0], dtype=torch.bool, device=x0.device)
    budget = _selector_budget_from_length(int(valid_mask.sum().item()), selector_target_ratio)
    observed = [0, max(0, x0.shape[0] - 1)]
    target = torch.zeros(x0.shape[0], dtype=torch.float32, device=x0.device)
    target[observed] = 1.0

    baseline_margin = _oracle_retrieval_margin_for_observed_indices(
        oracle_model,
        oracle_diff,
        eval_wrapper,
        x0,
        valid_mask,
        cond,
        observed,
        positive_text_embedding,
        negative_text_embeddings,
        oracle_timesteps,
    )

    while len(observed) < budget:
        candidates = [idx for idx in range(1, x0.shape[0] - 1) if idx not in observed]
        if not candidates:
            break
        candidate_margins = _oracle_candidate_retrieval_margins(
            oracle_model,
            oracle_diff,
            eval_wrapper,
            x0,
            valid_mask,
            cond,
            observed,
            candidates,
            positive_text_embedding,
            negative_text_embeddings,
            oracle_timesteps,
        )
        gains = candidate_margins - baseline_margin
        best_pos = int(torch.argmax(gains).item())
        best_idx = candidates[best_pos]
        best_gain = float(max(gains[best_pos].item(), 0.0))
        observed.append(best_idx)
        baseline_margin = float(candidate_margins[best_pos].item())
        target[best_idx] = max(target[best_idx].item(), best_gain)

    target = target * valid_mask.float()
    max_val = target.max().clamp(min=1e-6)
    target = target / max_val
    target[0] = 1.0
    target[max(0, x0.shape[0] - 1)] = 1.0
    return target.cpu()


def _prepare_information_gain_oracle_targets(dataset, cfg, device: torch.device, oracle_ckpt_path: str):
    cache_path = _selector_oracle_cache_path(cfg, dataset.split, oracle_ckpt_path)
    if os.path.exists(cache_path):
        print(f'Loading information-gain oracle cache: {cache_path}')
        return torch.load(cache_path)

    oracle_model, oracle_diff = _load_oracle_inbetween_model(oracle_ckpt_path, dataset.feature_dim, device)
    oracle_timesteps_count = max(1, int(getattr(cfg, 'selector_oracle_timesteps', 3)))
    oracle_timesteps = torch.linspace(1, cfg.T_diffusion - 1, steps=oracle_timesteps_count).round().long().tolist()

    oracle_targets = {}
    print(f'Building information-gain oracle cache for split={dataset.split} using {oracle_ckpt_path}')
    for idx, item in enumerate(dataset.data, start=1):
        motion = item['motion'].to(device)
        if dataset.normalize:
            motion = (motion - dataset.mean.to(device)) / (dataset.std.to(device) + 1e-8)
        cond = dataset.embeddings[item['id']].mean(dim=0, keepdim=True).to(device)
        oracle_targets[item['id']] = _build_information_gain_target_for_item(
            oracle_model,
            oracle_diff,
            motion,
            cond,
            cfg.selector_target_ratio,
            oracle_timesteps,
        )
        if idx % 100 == 0:
            print(f'  oracle labels: {idx}/{len(dataset.data)}')

    torch.save(oracle_targets, cache_path)
    print(f'Saved information-gain oracle cache: {cache_path}')
    return oracle_targets


def _prepare_retrieval_gain_oracle_targets(dataset, cfg, device: torch.device, oracle_ckpt_path: str):
    cache_path = _selector_oracle_cache_path(cfg, dataset.split, oracle_ckpt_path)
    if os.path.exists(cache_path):
        print(f'Loading retrieval-gain oracle cache: {cache_path}')
        return torch.load(cache_path)

    eval_root = cfg.selector_eval_root or _default_selector_eval_root()
    if not eval_root:
        raise FileNotFoundError(
            'Retrieval-gain selector requires a T2M-GPT eval root. '
            'Pass --selector-eval-root or place T2M-GPT adjacent to this repo.'
        )
    eval_root = os.path.abspath(eval_root)
    eval_wrapper, word_vectorizer = _load_retrieval_eval_components(eval_root, device)
    text_embedding_cache = _build_retrieval_text_embedding_cache(dataset, eval_wrapper, word_vectorizer)
    oracle_model, oracle_diff = _load_oracle_inbetween_model(oracle_ckpt_path, dataset.feature_dim, device)
    oracle_timesteps_count = max(1, int(getattr(cfg, 'selector_oracle_timesteps', 3)))
    oracle_timesteps = torch.linspace(1, cfg.T_diffusion - 1, steps=oracle_timesteps_count).round().long().tolist()
    negative_count = max(1, int(getattr(cfg, 'selector_retrieval_negatives', 31)))

    oracle_targets = {}
    print(f'Building retrieval-gain oracle cache for split={dataset.split} using {oracle_ckpt_path}')
    for idx, item in enumerate(dataset.data, start=1):
        motion = item['motion'].to(device)
        if dataset.normalize:
            motion = (motion - dataset.mean.to(device)) / (dataset.std.to(device) + 1e-8)
        cond = dataset.embeddings[item['id']].mean(dim=0, keepdim=True).to(device)
        positive_text_embedding = text_embedding_cache[item['id']].mean(dim=0).to(device)
        negative_text_embeddings = _sample_retrieval_negative_embeddings(
            item['id'],
            text_embedding_cache,
            negative_count,
        ).to(device)
        oracle_targets[item['id']] = _build_retrieval_gain_target_for_item(
            oracle_model,
            oracle_diff,
            eval_wrapper,
            motion,
            cond,
            positive_text_embedding,
            negative_text_embeddings,
            cfg.selector_target_ratio,
            oracle_timesteps,
        )
        if idx % 100 == 0:
            print(f'  oracle labels: {idx}/{len(dataset.data)}')

    torch.save(oracle_targets, cache_path)
    print(f'Saved retrieval-gain oracle cache: {cache_path}')
    return oracle_targets


def _selector_mode_uses_oracle_targets(selector_mode: str) -> bool:
    return str(selector_mode).strip().lower() in {'information_gain', 'retrieval_gain'}


def _prepare_selector_oracle_targets(dataset, cfg, device: torch.device, oracle_ckpt_path: str):
    mode = str(cfg.selector_mode).strip().lower()
    if mode == 'information_gain':
        return _prepare_information_gain_oracle_targets(dataset, cfg, device, oracle_ckpt_path)
    if mode == 'retrieval_gain':
        return _prepare_retrieval_gain_oracle_targets(dataset, cfg, device, oracle_ckpt_path)
    raise ValueError(f'Selector mode {mode!r} does not use oracle targets.')


def _gather_selector_oracle_targets(batch, oracle_targets: dict | None, device: torch.device):
    if not oracle_targets:
        return None
    if any(sample_id not in oracle_targets for sample_id in batch['ids']):
        return None

    lengths = batch['lengths']
    T_max = int(lengths.max().item())
    out = torch.zeros(len(batch['ids']), T_max, dtype=torch.float32, device=device)
    for i, sample_id in enumerate(batch['ids']):
        target = oracle_targets[sample_id].to(device)
        out[i, :target.shape[0]] = target[:T_max]
    return out


def _default_inbetween_ckpt_prefix(cfg) -> str:
    if not cfg.use_learned_keyframe_selector:
        return 'composite_inbetween_dataset'
    selector_mode = str(cfg.selector_mode).strip().lower().replace('-', '_')
    return f'composite_inbetween_{selector_mode}'


def _init_metrics_logger(stage_name: str, columns: list[str]):
    os.makedirs('training_logs', exist_ok=True)
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join('training_logs', f'{stage_name}_{run_stamp}.csv')
    plot_path = os.path.join('training_logs', f'{stage_name}_{run_stamp}.png')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

    return {
        'stage_name': stage_name,
        'columns': columns,
        'csv_path': csv_path,
        'plot_path': plot_path,
        'rows': [],
    }


def _append_metrics_row(logger, row: dict):
    out = {col: row.get(col, '') for col in logger['columns']}
    with open(logger['csv_path'], 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=logger['columns'])
        writer.writerow(out)
    logger['rows'].append(out)


def _save_convergence_plot(logger):
    if plt is None:
        return

    if not logger['rows']:
        return

    numeric_cols = [c for c in logger['columns'] if c != 'step']
    if not numeric_cols:
        return

    series = {c: [] for c in numeric_cols}
    steps = []
    for row in logger['rows']:
        step = _safe_float(row.get('step'))
        if step is None:
            continue

        has_any = False
        parsed = {}
        for col in numeric_cols:
            val = _safe_float(row.get(col))
            parsed[col] = val
            if val is not None:
                has_any = True

        if not has_any:
            continue

        steps.append(step)
        for col in numeric_cols:
            series[col].append(parsed[col])

    if not steps:
        return

    plt.figure(figsize=(10, 6))
    for col in numeric_cols:
        col_vals = series[col]
        if not col_vals or all(v is None for v in col_vals):
            continue
        ys = [np.nan if v is None else v for v in col_vals]
        plt.plot(steps, ys, label=col)

    plt.title(f"{logger['stage_name']} convergence")
    plt.xlabel('step')
    plt.ylabel('metric value')
    plt.grid(alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(logger['plot_path'], dpi=160)
    plt.close()


def _lr_scale(step: int, total_steps: int, warmup_ratio: float, min_lr_ratio: float, scheduler_type: str) -> float:
    if scheduler_type == 'constant' or total_steps <= 0:
        return 1.0

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step <= warmup_steps:
        return max(1e-8, float(step) / float(warmup_steps))

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _apply_lr_schedule(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    step: int,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
    scheduler_type: str,
):
    scale = _lr_scale(step, total_steps, warmup_ratio, min_lr_ratio, scheduler_type)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg['lr'] = float(base_lr) * scale


def _clone_state_dict(module: nn.Module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _update_ema_state(ema_state: dict[str, torch.Tensor], module: nn.Module, decay: float):
    with torch.no_grad():
        current = module.state_dict()
        for k, v in current.items():
            if k not in ema_state:
                ema_state[k] = v.detach().clone()
                continue
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.detach(), alpha=(1.0 - decay))
            else:
                ema_state[k].copy_(v)


def _build_cond(batch, dataset, empty_emb, p_uncond: float):
    cond_list = []
    for mid, tidx in zip(batch['ids'], batch['text_idxs']):
        if random.random() < p_uncond:
            cond_list.append(empty_emb)
        else:
            cond_list.append(dataset.get_embedding(mid, tidx))
    return torch.stack(cond_list)


def _compute_inbetween_loss(
    inbetween_model,
    diff_inbetween,
    x0,
    mask,
    keyframes,
    keyframe_indices,
    keyframe_mask,
    cond,
    cfg,
    keyframe_selector,
    selector_budget_weight: float,
    selector_entropy_weight: float,
    selector_oracle_target: torch.Tensor | None = None,
):
    B = x0.shape[0]
    t = torch.randint(0, cfg.T_diffusion, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = diff_inbetween.q_sample(x0, t, noise)
    xt = xt * mask.float().unsqueeze(-1)

    selector_ratio = None
    selector_budget_loss = x0.new_tensor(0.0)
    selector_entropy_loss = x0.new_tensor(0.0)
    selector_aux_loss = x0.new_tensor(0.0)

    if keyframe_selector is not None:
        selector_is_trainable = getattr(keyframe_selector, 'is_trainable', True)
        selector_probs, selector_mask_st = keyframe_selector(x0, mask, cond=cond)
        keyframe_canvas = selector_mask_st.unsqueeze(-1) * x0

        xt = diff_inbetween._replace_keyframes(
            xt,
            observation_mask=selector_mask_st,
            keyframe_canvas=keyframe_canvas,
        )
        x0_hat = inbetween_model(
            xt,
            t,
            cond,
            mask,
            observation_mask=selector_mask_st,
            keyframe_canvas=keyframe_canvas,
        )

        non_keyframe_weights = (1.0 - selector_mask_st) * mask.float()
        loss = weighted_masked_mse(x0, x0_hat, non_keyframe_weights, mask)

        selector_ratio = (selector_probs * mask.float()).sum() / (mask.float().sum() + 1e-8)
        if selector_is_trainable:
            selector_budget_loss = (selector_ratio - cfg.selector_target_ratio) ** 2
            p = selector_probs.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            selector_entropy_loss = (entropy * mask.float()).sum() / (mask.float().sum() + 1e-8)
            selector_aux_loss = keyframe_selector.compute_auxiliary_loss(
                x0,
                mask,
                cond,
                selector_probs,
                selector_mask_st,
                oracle_target=selector_oracle_target,
            )
        loss = (
            loss
            + selector_budget_weight * selector_budget_loss
            + selector_entropy_weight * selector_entropy_loss
            + float(cfg.selector_aux_weight) * selector_aux_loss
        )

        keyframes, keyframe_indices, keyframe_mask = _frame_mask_to_sparse_keyframes(
            x0,
            selector_mask_st.detach(),
            mask,
        )
    else:
        xt = diff_inbetween._replace_keyframes(xt, keyframes, keyframe_indices, keyframe_mask)
        x0_hat = inbetween_model(xt, t, cond, mask, keyframes, keyframe_indices, keyframe_mask)
        non_keyframe_mask = mask.clone()
        for b in range(B):
            valid = keyframe_mask[b]
            valid_idx = keyframe_indices[b][valid]
            non_keyframe_mask[b, valid_idx] = False
        loss = masked_mse(x0, x0_hat, non_keyframe_mask)

    boundary_jerk_weight = float(getattr(cfg, 'boundary_jerk_weight', 0.0))
    if boundary_jerk_weight > 0.0:
        loss = loss + boundary_jerk_weight * boundary_jerk_loss(
            x0,
            x0_hat,
            mask,
            keyframe_indices,
            keyframe_mask,
        )

    return {
        'loss': loss,
        'selector_ratio': selector_ratio,
        'selector_budget': selector_budget_loss,
        'selector_entropy': selector_entropy_loss,
        'selector_aux': selector_aux_loss,
    }


@torch.no_grad()
def _evaluate_inbetween(
    inbetween_model,
    keyframe_selector,
    diff_inbetween,
    val_loader,
    dataset,
    empty_emb,
    cfg,
    device,
    selector_oracle_targets: dict | None = None,
):
    inbetween_model.eval()
    if keyframe_selector is not None:
        keyframe_selector.eval()

    loss_values = []
    max_batches = max(1, int(cfg.val_batches))
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x0 = batch['motion'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        keyframes = batch['keyframes'].to(device, non_blocking=True)
        keyframe_indices = batch['keyframe_indices'].to(device, non_blocking=True)
        keyframe_mask = batch['keyframe_mask'].to(device, non_blocking=True)
        cond = _build_cond(batch, dataset, empty_emb, p_uncond=0.0)
        selector_oracle_target = _gather_selector_oracle_targets(batch, selector_oracle_targets, device)

        out = _compute_inbetween_loss(
            inbetween_model,
            diff_inbetween,
            x0,
            mask,
            keyframes,
            keyframe_indices,
            keyframe_mask,
            cond,
            cfg,
            keyframe_selector,
            selector_budget_weight=cfg.selector_budget_weight,
            selector_entropy_weight=cfg.selector_entropy_weight,
            selector_oracle_target=selector_oracle_target,
        )
        loss_values.append(out['loss'].item())

    inbetween_model.train()
    if keyframe_selector is not None:
        keyframe_selector.train()

    if not loss_values:
        return None
    return float(sum(loss_values) / len(loss_values))


def main(
    stage: str,
    force_retrain: bool,
    vqvae_steps: int | None = None,
    gpt_steps: int | None = None,
    inbetween_steps: int | None = None,
    inbetween_resume: str | None = None,
    keyframe_source_dir: str | None = None,
    disable_selector: bool = False,
    lr: float | None = None,
    inbetween_ckpt_prefix: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    scheduler_type: str | None = None,
    warmup_ratio: float | None = None,
    min_lr_ratio: float | None = None,
    inbetween_lr: float | None = None,
    selector_lr: float | None = None,
    selector_mode: str | None = None,
    selector_oracle_ckpt: str | None = None,
    selector_oracle_timesteps: int | None = None,
    selector_eval_root: str | None = None,
    selector_retrieval_negatives: int | None = None,
    ema_decay: float | None = None,
    selector_curriculum_fraction: float | None = None,
    val_interval: int | None = None,
    val_batches: int | None = None,
    grad_accum_steps: int | None = None,
):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    cfg = CompositeConfig(
        keyframe_strategy='random',      # Switch to random
        keyframe_min=3,                  # Min keyframes per sequence
        keyframe_max=20,                 # Max keyframes per sequence
        keyframe_include_ends=True,      # Always include first/last frame
        # ... other params
    )
    if vqvae_steps is not None:
        cfg.vqvae_steps = int(vqvae_steps)
    if gpt_steps is not None:
        cfg.gpt_steps = int(gpt_steps)
    if inbetween_steps is not None:
        cfg.inbetween_steps = int(inbetween_steps)
    if inbetween_resume is not None:
        inbetween_resume = inbetween_resume.strip()
    if keyframe_source_dir is not None:
        cfg.keyframe_source_dir = keyframe_source_dir
    if disable_selector:
        cfg.use_learned_keyframe_selector = False
    if lr is not None:
        cfg.lr = float(lr)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if num_workers is not None:
        cfg.num_workers = int(num_workers)
    if scheduler_type is not None:
        cfg.scheduler_type = scheduler_type
    if warmup_ratio is not None:
        cfg.warmup_ratio = float(warmup_ratio)
    if min_lr_ratio is not None:
        cfg.min_lr_ratio = float(min_lr_ratio)
    if inbetween_lr is not None:
        cfg.inbetween_lr = float(inbetween_lr)
    if selector_lr is not None:
        cfg.selector_lr = float(selector_lr)
    if selector_mode is not None:
        cfg.selector_mode = selector_mode.strip().lower()
        if cfg.selector_mode not in SELECTOR_MODE_CHOICES:
            raise ValueError(f"Unknown selector mode {cfg.selector_mode!r}. Expected one of {SELECTOR_MODE_CHOICES}.")
    if selector_oracle_ckpt is not None:
        cfg.selector_oracle_ckpt_path = selector_oracle_ckpt.strip()
    if selector_oracle_timesteps is not None:
        cfg.selector_oracle_timesteps = max(1, int(selector_oracle_timesteps))
    if selector_eval_root is not None:
        cfg.selector_eval_root = selector_eval_root.strip()
    if selector_retrieval_negatives is not None:
        cfg.selector_retrieval_negatives = max(1, int(selector_retrieval_negatives))
    if ema_decay is not None:
        cfg.ema_decay = float(ema_decay)
    if selector_curriculum_fraction is not None:
        cfg.selector_curriculum_fraction = float(selector_curriculum_fraction)
    if val_interval is not None:
        cfg.val_interval = int(val_interval)
    if val_batches is not None:
        cfg.val_batches = int(val_batches)
    if grad_accum_steps is not None:
        cfg.grad_accum_steps = max(1, int(grad_accum_steps))
    if inbetween_ckpt_prefix is None or not inbetween_ckpt_prefix.strip():
        inbetween_ckpt_prefix = _default_inbetween_ckpt_prefix(cfg)
    inbetween_ckpt_prefix = inbetween_ckpt_prefix.strip()
    if stage in {'all', 'inbetween'}:
        selector_state = 'enabled' if cfg.use_learned_keyframe_selector else 'disabled'
        print(
            f"Keyframe strategy (dataset fallback): {cfg.keyframe_strategy} | "
            f"selector: {selector_state} ({cfg.selector_mode})"
        )
    else:
        print(f"Keyframe strategy: {cfg.keyframe_strategy}")

    inbetween_final_ckpt_path = f'checkpoints/{inbetween_ckpt_prefix}_step{cfg.inbetween_steps}.pt'
    if stage in {'all', 'inbetween'} and not force_retrain:
        arch_ckpt_path = None
        if inbetween_resume is not None and os.path.exists(inbetween_resume):
            arch_ckpt_path = inbetween_resume
        elif os.path.exists(inbetween_final_ckpt_path):
            arch_ckpt_path = inbetween_final_ckpt_path
        if arch_ckpt_path is not None:
            _apply_inbetween_arch_from_checkpoint(cfg, arch_ckpt_path, device)
            print(
                'In-between architecture from checkpoint:',
                arch_ckpt_path,
                f"d_model={cfg.inbetween_d_model}",
                f"layers={cfg.inbetween_layers}",
                f"selector_d_model={cfg.selector_d_model}",
                f"selector_layers={cfg.selector_layers}",
                f"selector_mode={cfg.selector_mode}",
            )
    
    # Load normalization statistics
    mean_path = os.path.join(cfg.root, 'Mean.npy')
    std_path = os.path.join(cfg.root, 'Std.npy')
    mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
    std = torch.from_numpy(np.load(std_path)).float().view(-1)
    Fdim = mean.shape[0]
    print('Feature dim:', Fdim)
    
    # Setup CLIP
    clip_model = setup_clip_model(device)
    
    text_encoder = TextEncoderWrapper(clip_model)
    
    empty_emb, empty_token_features, empty_token_mask = text_encoder.encode_with_tokens([''])
    empty_emb = empty_emb.squeeze(0)
    empty_token_features = empty_token_features.squeeze(0)
    empty_token_mask = empty_token_mask.squeeze(0)
    print('Empty embedding norm:', empty_emb.norm().item())
    
    # Create dataset
    dataset = HUMANML3DCompositeDataset(
        cfg.root,
        split='train',
        max_len=cfg.max_len,
        normalize=True,
        use_cache=True,
        text_encoder=text_encoder,
        keyframe_interval=cfg.keyframe_interval,
        keyframe_strategy=cfg.keyframe_strategy,
        keyframe_count=cfg.keyframe_count,
        keyframe_min=cfg.keyframe_min,
        keyframe_max=cfg.keyframe_max,
        keyframe_include_ends=cfg.keyframe_include_ends,
        include_keyframes=(stage != 'gpt'),
        conditioning_motion_dir=cfg.keyframe_source_dir,
        mean=mean,
        std=std,
        device=device,
    )
    print('Dataset size:', len(dataset))

    val_dataset = None
    val_loader = None
    if stage in {'all', 'inbetween'}:
        val_split_file = os.path.join(cfg.root, f'{cfg.val_split}.txt')
        if os.path.exists(val_split_file):
            val_dataset = HUMANML3DCompositeDataset(
                cfg.root,
                split=cfg.val_split,
                max_len=cfg.max_len,
                normalize=True,
                use_cache=True,
                text_encoder=text_encoder,
                keyframe_interval=cfg.keyframe_interval,
                keyframe_strategy=cfg.keyframe_strategy,
                keyframe_count=cfg.keyframe_count,
                keyframe_min=cfg.keyframe_min,
                keyframe_max=cfg.keyframe_max,
                keyframe_include_ends=cfg.keyframe_include_ends,
                conditioning_motion_dir=cfg.keyframe_source_dir,
                mean=mean,
                std=std,
                device=device,
            )
            val_loader_kwargs = {
                'batch_size': min(cfg.val_batch_size, cfg.batch_size),
                'shuffle': False,
                'num_workers': cfg.num_workers,
                'collate_fn': collate_composite,
                'drop_last': False,
                'pin_memory': bool(cfg.pin_memory and device.type == 'cuda'),
            }
            if cfg.num_workers > 0:
                val_loader_kwargs['persistent_workers'] = cfg.persistent_workers
                val_loader_kwargs['prefetch_factor'] = cfg.prefetch_factor
            val_loader = DataLoader(val_dataset, **val_loader_kwargs)
            print(f"Validation dataset size: {len(val_dataset)} ({cfg.val_split})")
        else:
            print(f"Validation split file not found: {val_split_file}; best-checkpoint selection disabled.")

    selector_oracle_targets = None
    val_selector_oracle_targets = None
    if stage in {'all', 'inbetween'} and cfg.use_learned_keyframe_selector and _selector_mode_uses_oracle_targets(cfg.selector_mode):
        oracle_ckpt_path = cfg.selector_oracle_ckpt_path or _default_selector_oracle_ckpt_path()
        if not oracle_ckpt_path:
            raise FileNotFoundError(
                f'{cfg.selector_mode} selector requires an oracle diffusion checkpoint. '
                'Pass --selector-oracle-ckpt or train a transformer in-between checkpoint first.'
            )
        oracle_ckpt_path = os.path.abspath(oracle_ckpt_path)
        selector_oracle_targets = _prepare_selector_oracle_targets(dataset, cfg, device, oracle_ckpt_path)
        if val_dataset is not None:
            val_selector_oracle_targets = _prepare_selector_oracle_targets(val_dataset, cfg, device, oracle_ckpt_path)
    
    loader_kwargs = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'collate_fn': collate_composite,
        'drop_last': True,
        'pin_memory': bool(cfg.pin_memory and device.type == 'cuda'),
    }
    if cfg.num_workers > 0:
        loader_kwargs['persistent_workers'] = cfg.persistent_workers
        loader_kwargs['prefetch_factor'] = cfg.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    print(
        'DataLoader:',
        f"workers={cfg.num_workers}",
        f"batch_size={cfg.batch_size}",
        f"grad_accum={cfg.grad_accum_steps}",
        f"effective_batch={cfg.batch_size * cfg.grad_accum_steps}",
        f"pin_memory={loader_kwargs['pin_memory']}",
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)}",
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}"
    )
    
    # Initialize only models needed for selected stage(s).
    vqvae = None
    gpt = None
    if stage in {'all', 'vqvae', 'gpt'}:
        vqvae = MotionVQVAE(
            feature_dim=Fdim,
            codebook_size=cfg.codebook_size,
            codebook_dim=cfg.codebook_dim,
            downsample_rate=cfg.downsample_rate,
            commitment_cost=cfg.commitment_cost,
        ).to(device)
        print(f'VQ-VAE params: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M')

    if stage in {'all', 'gpt'}:
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
    elif stage == 'inbetween':
        print('Skipping VQ-VAE and MotionGPT initialization (stage=inbetween).')
    
    inbetween_model = InbetweenTransformer(
        feature_dim=Fdim,
        cond_dim=512,
        d_model=cfg.inbetween_d_model,
        n_layers=cfg.inbetween_layers,
        n_heads=cfg.inbetween_heads,
        dropout=cfg.dropout,
        max_len=cfg.max_len + 10,
    ).to(device)
    print(f'In-betweening model params: {sum(p.numel() for p in inbetween_model.parameters())/1e6:.2f}M')

    keyframe_selector = None
    if cfg.use_learned_keyframe_selector:
        keyframe_selector = build_keyframe_selector(
            mode=cfg.selector_mode,
            feature_dim=Fdim,
            cond_dim=512,
            d_model=cfg.selector_d_model,
            n_layers=cfg.selector_layers,
            n_heads=cfg.selector_heads,
            dropout=cfg.selector_dropout,
            max_len=cfg.max_len + 10,
            threshold=cfg.selector_threshold,
            budget_ratio=cfg.selector_target_ratio,
        ).to(device)
        print(
            f"Keyframe selector [{cfg.selector_mode}] params: "
            f"{sum(p.numel() for p in keyframe_selector.parameters())/1e6:.2f}M"
        )
    
    diff_inbetween = InbetweenDiffusion(cfg.T_diffusion, device=device)
    
    # Stage 1a: Train VQ-VAE
    if stage in {'all', 'vqvae'}:
        if vqvae is None:
            raise RuntimeError('VQ-VAE model is not initialized for requested stage.')
        train_vqvae(vqvae, loader, cfg, device, mean, std, force_retrain)
    
    # Stage 1b: Train GPT
    if stage in {'all', 'gpt'}:
        
        if vqvae is None or gpt is None:
            raise RuntimeError('VQ-VAE/GPT models are not initialized for requested stage.')
        
        vqvae_ckpt_path = f'checkpoints/composite_vqvae_step{cfg.vqvae_steps}.pt'
        if not os.path.exists(vqvae_ckpt_path):
            raise FileNotFoundError(f"Missing VQ-VAE checkpoint: {vqvae_ckpt_path}")

        print(f"Loading trained VQ-VAE for GPT tokenisation: {vqvae_ckpt_path}")
        vqvae_ckpt = torch.load(vqvae_ckpt_path, map_location=device)
        vqvae.load_state_dict(vqvae_ckpt['model'])
        vqvae.eval()
        
        train_gpt(
            vqvae,
            gpt,
            loader,
            dataset,
            empty_token_features,
            empty_token_mask,
            cfg,
            device,
            force_retrain,
        )
    
    # Stage 2: Train Diffusion In-betweening
    if stage in {'all', 'inbetween'}:
        train_inbetween(
            inbetween_model,
            diff_inbetween,
            loader,
            val_loader,
            dataset,
            empty_emb,
            cfg,
            device,
            mean,
            std,
            force_retrain,
            keyframe_selector=keyframe_selector,
            resume_ckpt_path=inbetween_resume,
            ckpt_prefix=inbetween_ckpt_prefix,
            selector_oracle_targets=selector_oracle_targets,
            val_selector_oracle_targets=val_selector_oracle_targets,
        )
    
    print("Training complete!")


def train_vqvae(vqvae, loader, cfg, device, mean, std, force_retrain: bool = False):
    """Stage 1a: Train VQ-VAE."""
    vqvae_final_ckpt_path = f'checkpoints/composite_vqvae_step{cfg.vqvae_steps}.pt'
    if os.path.exists(vqvae_final_ckpt_path) and not force_retrain:
        print(f"Loading VQ-VAE from final checkpoint: {vqvae_final_ckpt_path}")
        vqvae_ckpt = torch.load(vqvae_final_ckpt_path, map_location=device)
        vqvae.load_state_dict(vqvae_ckpt['model'])
        print("VQ-VAE training already complete! Loaded from checkpoint.")
        return
    
    vqvae_optimizer = torch.optim.AdamW(vqvae.parameters(), lr=cfg.lr)
    base_lrs = [cfg.lr]
    lambda_recon = 1.0
    lambda_vel = 0.5
    metrics_logger = _init_metrics_logger(
        'vqvae',
        ['step', 'recon_loss', 'vq_loss', 'vel_loss', 'total_loss']
    )
    print(f"VQ-VAE metrics CSV: {metrics_logger['csv_path']}")
    
    vqvae.train()
    data_iter = cycle(loader)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    
    print("Stage 1a: Training VQ-VAE...")
    start_time = time.time()
    
    for step in range(1, cfg.vqvae_steps + 1):
        _apply_lr_schedule(
            vqvae_optimizer,
            base_lrs,
            step,
            cfg.vqvae_steps,
            cfg.warmup_ratio,
            cfg.min_lr_ratio,
            cfg.scheduler_type,
        )
        vqvae_optimizer.zero_grad(set_to_none=True)
        recon_running = 0.0
        vq_running = 0.0
        vel_running = 0.0
        total_running = 0.0

        for _ in range(grad_accum_steps):
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
            (loss / grad_accum_steps).backward()

            recon_running += recon_loss.item()
            vq_running += vq_loss.item()
            vel_running += v_loss.item()
            total_running += loss.item()

        nn.utils.clip_grad_norm_(vqvae.parameters(), cfg.grad_clip)
        vqvae_optimizer.step()

        recon_avg = recon_running / grad_accum_steps
        vq_avg = vq_running / grad_accum_steps
        vel_avg = vel_running / grad_accum_steps
        total_avg = total_running / grad_accum_steps
        
        if step % 200 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            steps_done = step
            steps_remaining = cfg.vqvae_steps - step
            avg_time_per_step = elapsed / steps_done
            eta_seconds = steps_remaining * avg_time_per_step
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            lr_now = vqvae_optimizer.param_groups[0]['lr']
            print(f"step {step:>7d} | recon {recon_avg:.5f} | vq {vq_avg:.5f} | vel {vel_avg:.5f} | lr {lr_now:.2e} | ETA: {eta_hours}h {eta_minutes}m")
            _append_metrics_row(metrics_logger, {
                'step': step,
                'recon_loss': recon_avg,
                'vq_loss': vq_avg,
                'vel_loss': vel_avg,
                'total_loss': total_avg,
            })

        if step % 2_000 == 0:
            _save_convergence_plot(metrics_logger)
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model': vqvae.state_dict(),
                'optimizer': vqvae_optimizer.state_dict(),
                'step': step,
            }, f'checkpoints/composite_vqvae_step{step}.pt')
            print('Saved VQ-VAE checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model': vqvae.state_dict(),
        'optimizer': vqvae_optimizer.state_dict(),
        'step': cfg.vqvae_steps,
    }, vqvae_final_ckpt_path)
    print(f'Saved final VQ-VAE checkpoint: {vqvae_final_ckpt_path}')
    _save_convergence_plot(metrics_logger)
    print(f"VQ-VAE convergence plot: {metrics_logger['plot_path']}")
    
    print("VQ-VAE training complete!")


# ---------------------------------------------------------------------------
# GPT pre-tokenization helpers
# ---------------------------------------------------------------------------

def _pretokenize_dataset(vqvae, dataset, cfg, device):
    """Run VQ-VAE encode over every dataset motion once and cache the tokens.

    VQ tokens are deterministic (argmax), so running the encoder on every GPT
    training step is pure overhead.  This function computes the tokens once,
    saves them to disk, and returns a dict ``{sample_id: (tokens_cpu, token_len)}``.
    """
    cache_path = os.path.join(
        cfg.root,
        f'gpt_tokens_cb{cfg.codebook_size}_ds{cfg.downsample_rate}_maxlen{cfg.max_len}.pt',
    )
    if os.path.exists(cache_path):
        print(f'Loading pre-tokenised GPT data from {cache_path}')
        return torch.load(cache_path, map_location='cpu')

    print('Pre-tokenising dataset with VQ-VAE (one-time cost, result cached)...')
    vqvae.eval()
    tokens_dict = {}
    n = len(dataset.data)
    with torch.no_grad():
        for i, item in enumerate(dataset.data):
            mid = item['id']
            motion = item['motion']          # (T, F) normalised CPU tensor
            actual_length = int(item['length'])
            T = motion.shape[0]

            x = motion.unsqueeze(0).to(device)
            pad_len = (cfg.downsample_rate - T % cfg.downsample_rate) % cfg.downsample_rate
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))

            indices, _ = vqvae.encode(x)          # (1, T_tokens)
            tok_len = min(
                (actual_length + cfg.downsample_rate - 1) // cfg.downsample_rate,
                indices.shape[1],
            )
            tokens_dict[mid] = (indices[0].cpu(), tok_len)

            if (i + 1) % 1000 == 0 or (i + 1) == n:
                print(f'  tokenised {i + 1}/{n} sequences')

    torch.save(tokens_dict, cache_path)
    print(f'Saved token cache: {cache_path}')
    return tokens_dict


def _build_gpt_batch_from_tokens(batch, tokens_dict, gpt, device):
    """Build GPT input/target/mask tensors from pre-tokenised data.

    Replaces ``prepare_gpt_batch`` in the hot loop, eliminating the per-step
    VQ-VAE encode.
    """
    ids = batch['ids']
    B = len(ids)

    token_seqs = []
    token_lengths = []
    for mid in ids:
        tokens, tok_len = tokens_dict[mid]
        token_seqs.append(tokens[:tok_len])
        token_lengths.append(tok_len)

    T_max = max(token_lengths)

    input_tokens = torch.full(
        (B, T_max + 1),
        gpt.pad_token,
        dtype=torch.long,
        device=device,
    )
    target_tokens = torch.full(
        (B, T_max + 1),
        gpt.pad_token,
        dtype=torch.long,
        device=device,
    )
    target_mask = torch.zeros(
        B,
        T_max + 1,
        dtype=torch.bool,
        device=device,
    )

    for i, (seq, tlen) in enumerate(zip(token_seqs, token_lengths)):
        seq = seq[:tlen].to(device, non_blocking=True)

        input_tokens[i, 0] = gpt.bos_token
        input_tokens[i, 1:tlen + 1] = seq

        target_tokens[i, :tlen] = seq
        target_tokens[i, tlen] = gpt.eos_token

        target_mask[i, :tlen + 1] = True

    return input_tokens, target_tokens, target_mask


def _build_gpt_cond_batch(batch, dataset, empty_tokens, empty_mask, p_uncond: float, device):
    cond_list = []
    mask_list = []
    for mid, tidx in zip(batch['ids'], batch['text_idxs']):
        if random.random() < p_uncond:
            cond_list.append(empty_tokens)
            mask_list.append(empty_mask)
        else:
            cond_list.append(dataset.get_token_embedding(mid, tidx))
            mask_list.append(dataset.get_token_mask(mid, tidx))

    max_ctx = max(int(mask.shape[0]) for mask in mask_list)
    feat_dim = int(cond_list[0].shape[-1])
    cond = torch.zeros(len(cond_list), max_ctx, feat_dim, dtype=cond_list[0].dtype, device=device)
    cond_mask = torch.zeros(len(cond_list), max_ctx, dtype=torch.bool, device=device)

    for i, (feat, mask) in enumerate(zip(cond_list, mask_list)):
        ctx_len = int(mask.shape[0])
        cond[i, :ctx_len] = feat.to(device, non_blocking=True)
        cond_mask[i, :ctx_len] = mask.to(device, non_blocking=True)

    return cond, cond_mask


def train_gpt(vqvae, gpt, loader, dataset, empty_cond_tokens, empty_cond_mask, cfg, device, force_retrain: bool = False):
    """Stage 1b: Train MotionGPT."""
    gpt_final_ckpt_path = f'checkpoints/composite_gpt_step{cfg.gpt_steps}.pt'
    if os.path.exists(gpt_final_ckpt_path) and not force_retrain:
        print(f"Loading GPT from final checkpoint: {gpt_final_ckpt_path}")
        gpt_ckpt = torch.load(gpt_final_ckpt_path, map_location=device)
        try:
            gpt.load_state_dict(gpt_ckpt['gpt'])
        except RuntimeError as exc:
            raise RuntimeError(
                'Existing GPT checkpoint is incompatible with the new token-level text conditioning. '
                'Retrain GPT with --force to use the updated architecture.'
            ) from exc
        vqvae.load_state_dict(gpt_ckpt['vqvae'])
        print("MotionGPT training already complete! Loaded from checkpoint.")
        return
    
    vqvae.eval()
    gpt_optimizer = torch.optim.AdamW(gpt.parameters(), lr=cfg.lr)
    base_lrs = [cfg.lr]
    metrics_logger = _init_metrics_logger('gpt', ['step', 'loss', 'ppl'])
    print(f"GPT metrics CSV: {metrics_logger['csv_path']}")

    # Pre-tokenise the entire dataset once so the VQ-VAE encode is not
    # repeated on every training step (it's deterministic, so pure overhead).
    tokens_dict = _pretokenize_dataset(vqvae, dataset, cfg, device)

    gpt.train()
    data_iter = cycle(loader)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    
    print("Stage 1b: Training MotionGPT...")
    start_time = time.time()
    
    for step in range(1, cfg.gpt_steps + 1):
        _apply_lr_schedule(
            gpt_optimizer,
            base_lrs,
            step,
            cfg.gpt_steps,
            cfg.warmup_ratio,
            cfg.min_lr_ratio,
            cfg.scheduler_type,
        )
        gpt_optimizer.zero_grad(set_to_none=True)
        loss_running = 0.0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)

            input_tokens, target_tokens, target_mask = _build_gpt_batch_from_tokens(
                batch, tokens_dict, gpt, device
            )

            cond, cond_mask = _build_gpt_cond_batch(
                batch,
                dataset,
                empty_cond_tokens,
                empty_cond_mask,
                cfg.p_uncond,
                device,
            )

            logits = gpt(input_tokens, cond, cond_mask=cond_mask)

            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                target_tokens.reshape(-1),
                ignore_index=gpt.pad_token,
                reduction='none'
            ).reshape(B, T)

            loss = (loss * target_mask.float()).sum() / (target_mask.float().sum() + 1e-8)
            (loss / grad_accum_steps).backward()
            loss_running += loss.item()

        nn.utils.clip_grad_norm_(gpt.parameters(), cfg.grad_clip)
        gpt_optimizer.step()

        loss_avg = loss_running / grad_accum_steps
        
        if step % 200 == 0:
            ppl = math.exp(loss_avg)
            current_time = time.time()
            elapsed = current_time - start_time
            steps_remaining = cfg.gpt_steps - step
            avg_time_per_step = elapsed / step
            eta_seconds = steps_remaining * avg_time_per_step
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            lr_now = gpt_optimizer.param_groups[0]['lr']
            print(f"step {step:>7d} | loss {loss_avg:.5f} | ppl {ppl:.2f} | lr {lr_now:.2e} | ETA: {eta_hours}h {eta_minutes}m")
            _append_metrics_row(metrics_logger, {
                'step': step,
                'loss': loss_avg,
                'ppl': ppl,
            })

        if step % 2_000 == 0:
            _save_convergence_plot(metrics_logger)
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'gpt': gpt.state_dict(),
                'vqvae': vqvae.state_dict(),
                'optimizer': gpt_optimizer.state_dict(),
                'step': step,
            }, f'checkpoints/composite_gpt_step{step}.pt')
            print('Saved GPT checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'gpt': gpt.state_dict(),
        'vqvae': vqvae.state_dict(),
        'optimizer': gpt_optimizer.state_dict(),
        'step': cfg.gpt_steps,
    }, gpt_final_ckpt_path)
    print(f'Saved final GPT checkpoint: {gpt_final_ckpt_path}')
    _save_convergence_plot(metrics_logger)
    print(f"GPT convergence plot: {metrics_logger['plot_path']}")
    
    print("MotionGPT training complete!")


def _frame_mask_to_sparse_keyframes(
    x0: torch.Tensor,
    frame_mask: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """Convert per-frame keyframe mask (B, T) into padded sparse keyframe tensors (vectorized)."""
    B, T, F = x0.shape
    hard_mask = (frame_mask > 0.5) & valid_mask  # (B, T) bool

    # Count keyframes per batch item; pad all to K_max
    counts = hard_mask.sum(dim=1).clamp(min=1)  # (B,) — at least one per item
    K_max = int(counts.max().item())

    keyframe_indices = torch.zeros(B, K_max, device=x0.device, dtype=torch.long)
    keyframe_mask_out = torch.zeros(B, K_max, device=x0.device, dtype=torch.bool)

    # Fill row-by-row using the already-computed hard_mask; work entirely on CPU
    # for the index-packing step (one pass, no per-element GPU sync).
    hard_mask_cpu = hard_mask.cpu()
    for b in range(B):
        idx = hard_mask_cpu[b].nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.long)
        K = idx.numel()
        keyframe_indices[b, :K] = idx.to(x0.device)
        keyframe_mask_out[b, :K] = True

    # Gather keyframe features in one batched op (no loops)
    idx_exp = keyframe_indices.unsqueeze(-1).expand(B, K_max, F)  # (B, K_max, F)
    keyframes = torch.gather(x0, 1, idx_exp)                      # (B, K_max, F)

    return keyframes, keyframe_indices, keyframe_mask_out


def train_inbetween(
    inbetween_model,
    diff_inbetween,
    loader,
    val_loader,
    dataset,
    empty_emb,
    cfg,
    device,
    mean,
    std,
    force_retrain: bool = False,
    keyframe_selector=None,
    resume_ckpt_path: str | None = None,
    ckpt_prefix: str = 'composite_inbetween',
    selector_oracle_targets: dict | None = None,
    val_selector_oracle_targets: dict | None = None,
):
    """Stage 2: Train Diffusion In-betweening."""
    inbetween_final_ckpt_path = f'checkpoints/{ckpt_prefix}_step{cfg.inbetween_steps}.pt'
    inbetween_best_ckpt_path = f'checkpoints/{ckpt_prefix}_best.pt'
    if os.path.exists(inbetween_final_ckpt_path) and not force_retrain and not resume_ckpt_path:
        print(f"Loading In-Betweening model from final checkpoint: {inbetween_final_ckpt_path}")
        inbetween_ckpt = torch.load(inbetween_final_ckpt_path, map_location=device)
        if cfg.use_ema_for_sampling and inbetween_ckpt.get('inbetween_ema') is not None:
            inbetween_model.load_state_dict(inbetween_ckpt['inbetween_ema'])
        else:
            inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
        if keyframe_selector is not None:
            selector_key = 'selector_ema' if cfg.use_ema_for_sampling and inbetween_ckpt.get('selector_ema') is not None else 'selector'
            if inbetween_ckpt.get(selector_key) is not None:
                keyframe_selector.load_state_dict(inbetween_ckpt[selector_key])
            else:
                print('Warning: selector enabled in config but no selector weights found in checkpoint.')
        print("Diffusion in-betweening training already complete! Loaded from checkpoint.")
        return
    if os.path.exists(inbetween_final_ckpt_path) and not force_retrain and resume_ckpt_path:
        print(
            f"Ignoring existing final checkpoint {inbetween_final_ckpt_path} because explicit resume was requested: {resume_ckpt_path}"
        )

    inbetween_lr = float(cfg.inbetween_lr if cfg.inbetween_lr is not None else cfg.lr)
    selector_has_trainable_params = False
    selector_params = []
    if keyframe_selector is not None:
        selector_params = [p for p in keyframe_selector.parameters() if p.requires_grad]
        selector_has_trainable_params = len(selector_params) > 0 and getattr(keyframe_selector, 'is_trainable', True)

    if selector_has_trainable_params:
        selector_lr = float(cfg.selector_lr if cfg.selector_lr is not None else (inbetween_lr * cfg.selector_lr_scale))
        param_groups = [
            {'params': list(inbetween_model.parameters()), 'lr': inbetween_lr, 'name': 'inbetween'},
            {'params': selector_params, 'lr': selector_lr, 'name': 'selector'},
        ]
        base_lrs = [inbetween_lr, selector_lr]
    else:
        param_groups = [{'params': list(inbetween_model.parameters()), 'lr': inbetween_lr, 'name': 'inbetween'}]
        base_lrs = [inbetween_lr]

    inbetween_optimizer = torch.optim.AdamW(param_groups)
    all_params = []
    for group in param_groups:
        all_params.extend(group['params'])

    ema_inbetween_state = _clone_state_dict(inbetween_model)
    ema_selector_state = _clone_state_dict(keyframe_selector) if selector_has_trainable_params else None

    metrics_logger = _init_metrics_logger(
        'inbetween',
        [
            'step',
            'total_loss',
            'val_loss',
            'lr_inbetween',
            'lr_selector',
            'selector_budget_weight',
            'selector_entropy_weight',
            'selector_aux_weight',
            'selector_ratio',
            'selector_budget',
            'selector_entropy',
            'selector_aux',
        ],
    )
    print(f"In-between metrics CSV: {metrics_logger['csv_path']}")

    start_step = 0
    if resume_ckpt_path and os.path.exists(resume_ckpt_path) and not force_retrain:
        print(f"Resuming In-Betweening from checkpoint: {resume_ckpt_path}")
        inbetween_ckpt = torch.load(resume_ckpt_path, map_location=device)
        inbetween_model.load_state_dict(inbetween_ckpt['inbetween'])
        if inbetween_ckpt.get('inbetween_ema') is not None:
            ema_inbetween_state = {k: v.detach().clone() for k, v in inbetween_ckpt['inbetween_ema'].items()}
        if keyframe_selector is not None:
            if inbetween_ckpt.get('selector') is not None:
                keyframe_selector.load_state_dict(inbetween_ckpt['selector'])
                if selector_has_trainable_params and inbetween_ckpt.get('selector_ema') is not None:
                    ema_selector_state = {k: v.detach().clone() for k, v in inbetween_ckpt['selector_ema'].items()}
            else:
                print('Warning: resume checkpoint has no selector state; selector will train from scratch.')
        if 'optimizer' in inbetween_ckpt:
            try:
                inbetween_optimizer.load_state_dict(inbetween_ckpt['optimizer'])
            except Exception as exc:
                print(f"Warning: failed to load optimizer state, continuing with fresh optimizer ({exc})")
        start_step = int(inbetween_ckpt.get('step', 0))
        print(f"Resume step: {start_step}")
        if cfg.inbetween_steps <= start_step:
            print("Requested in-betweening steps already reached; nothing to train.")
            return

    best_val_loss = float('inf')
    best_step = 0
    if os.path.exists(inbetween_best_ckpt_path) and not force_retrain:
        try:
            best_ckpt = torch.load(inbetween_best_ckpt_path, map_location='cpu')
            best_val_loss = float(best_ckpt.get('best_val_loss', float('inf')))
            best_step = int(best_ckpt.get('best_step', 0))
            print(f"Current best checkpoint: step={best_step}, val_loss={best_val_loss:.6f}")
        except Exception as exc:
            print(f"Warning: failed to read best checkpoint metadata ({exc})")

    inbetween_model.train()
    if keyframe_selector is not None:
        keyframe_selector.train()
    data_iter = cycle(loader)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    
    if keyframe_selector is not None:
        print(
            f"Stage 2: Training Diffusion In-Betweening with selector={cfg.selector_mode} "
            f"(target ratio={cfg.selector_target_ratio})..."
        )
    else:
        print(f"Stage 2: Training Diffusion In-Betweening (keyframe interval={cfg.keyframe_interval})...")
    
    start_time = time.time()

    for step in range(start_step + 1, cfg.inbetween_steps + 1):
        _apply_lr_schedule(
            inbetween_optimizer,
            base_lrs,
            step,
            cfg.inbetween_steps,
            cfg.warmup_ratio,
            cfg.min_lr_ratio,
            cfg.scheduler_type,
        )
        inbetween_optimizer.zero_grad(set_to_none=True)

        if keyframe_selector is not None:
            curriculum_den = max(1.0, float(cfg.inbetween_steps) * float(cfg.selector_curriculum_fraction))
            curriculum_scale = min(1.0, max(0.0, float(step) / curriculum_den))
        else:
            curriculum_scale = 1.0

        selector_budget_weight = float(cfg.selector_budget_weight) * curriculum_scale
        selector_entropy_weight = float(cfg.selector_entropy_weight) * curriculum_scale

        total_loss_running = 0.0
        selector_ratio_running = 0.0
        selector_budget_running = 0.0
        selector_entropy_running = 0.0
        selector_aux_running = 0.0
        selector_metric_count = 0

        for _ in range(grad_accum_steps):
            batch = next(data_iter)
            x0 = batch['motion'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            keyframes = batch['keyframes'].to(device, non_blocking=True)
            keyframe_indices = batch['keyframe_indices'].to(device, non_blocking=True)
            keyframe_mask = batch['keyframe_mask'].to(device, non_blocking=True)
            selector_oracle_target = _gather_selector_oracle_targets(batch, selector_oracle_targets, device)

            cond = _build_cond(batch, dataset, empty_emb, p_uncond=cfg.p_uncond)

            out = _compute_inbetween_loss(
                inbetween_model,
                diff_inbetween,
                x0,
                mask,
                keyframes,
                keyframe_indices,
                keyframe_mask,
                cond,
                cfg,
                keyframe_selector,
                selector_budget_weight=selector_budget_weight,
                selector_entropy_weight=selector_entropy_weight,
                selector_oracle_target=selector_oracle_target,
            )
            loss = out['loss']
            (loss / grad_accum_steps).backward()

            total_loss_running += loss.item()
            if out['selector_ratio'] is not None:
                selector_metric_count += 1
                selector_ratio_running += out['selector_ratio'].item()
                selector_budget_running += out['selector_budget'].item()
                selector_entropy_running += out['selector_entropy'].item()
                selector_aux_running += out['selector_aux'].item()

        nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        inbetween_optimizer.step()

        total_loss_avg = total_loss_running / grad_accum_steps
        selector_ratio_avg = None
        selector_budget_avg = None
        selector_entropy_avg = None
        selector_aux_avg = None
        if selector_metric_count > 0:
            selector_ratio_avg = selector_ratio_running / selector_metric_count
            selector_budget_avg = selector_budget_running / selector_metric_count
            selector_entropy_avg = selector_entropy_running / selector_metric_count
            selector_aux_avg = selector_aux_running / selector_metric_count

        _update_ema_state(ema_inbetween_state, inbetween_model, cfg.ema_decay)
        if selector_has_trainable_params and keyframe_selector is not None and ema_selector_state is not None:
            _update_ema_state(ema_selector_state, keyframe_selector, cfg.ema_decay)

        val_loss = None
        if val_loader is not None and step % cfg.val_interval == 0:
            eval_inbetween_orig = _clone_state_dict(inbetween_model)
            eval_selector_orig = _clone_state_dict(keyframe_selector) if keyframe_selector is not None else None
            if cfg.use_ema_for_sampling:
                inbetween_model.load_state_dict(ema_inbetween_state)
                if selector_has_trainable_params and keyframe_selector is not None and ema_selector_state is not None:
                    keyframe_selector.load_state_dict(ema_selector_state)

            val_loss = _evaluate_inbetween(
                inbetween_model,
                keyframe_selector,
                diff_inbetween,
                val_loader,
                val_loader.dataset,
                empty_emb,
                cfg,
                device,
                selector_oracle_targets=val_selector_oracle_targets,
            )

            inbetween_model.load_state_dict(eval_inbetween_orig)
            if keyframe_selector is not None and eval_selector_orig is not None:
                keyframe_selector.load_state_dict(eval_selector_orig)

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'inbetween': inbetween_model.state_dict(),
                    'inbetween_ema': ema_inbetween_state,
                    'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
                    'selector_ema': ema_selector_state if selector_has_trainable_params else None,
                    'optimizer': inbetween_optimizer.state_dict(),
                    'step': step,
                    'best_step': best_step,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg.__dict__,
                }, inbetween_best_ckpt_path)
                print(f"Saved NEW BEST checkpoint at step {step} (val_loss={best_val_loss:.6f})")
        
        if step % 200 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            steps_done = step - start_step
            steps_remaining = cfg.inbetween_steps - step
            avg_time_per_step = elapsed / steps_done
            eta_seconds = steps_remaining * avg_time_per_step
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)

            lr_inbetween = inbetween_optimizer.param_groups[0]['lr']
            lr_selector = inbetween_optimizer.param_groups[1]['lr'] if len(inbetween_optimizer.param_groups) > 1 else None

            if selector_ratio_avg is None:
                print(
                    f"step {step:>7d} | loss {total_loss_avg:.5f} | "
                    f"lr {lr_inbetween:.2e} | "
                    f"val {val_loss if val_loss is not None else float('nan'):.5f} | "
                    f"ETA: {eta_hours}h {eta_minutes}m"
                )
            else:
                print(
                    f"step {step:>7d} | loss {total_loss_avg:.5f} | "
                    f"sel_ratio {selector_ratio_avg:.4f} | "
                    f"sel_budget {selector_budget_avg:.5f} | "
                    f"sel_entropy {selector_entropy_avg:.5f} | "
                    f"sel_aux {selector_aux_avg:.5f} | "
                    f"lr_d {lr_inbetween:.2e} | "
                    f"lr_s {lr_selector if lr_selector is not None else float('nan'):.2e} | "
                    f"val {val_loss if val_loss is not None else float('nan'):.5f} | "
                    f"ETA: {eta_hours}h {eta_minutes}m"
                )

            _append_metrics_row(metrics_logger, {
                'step': step,
                'total_loss': total_loss_avg,
                'val_loss': val_loss,
                'lr_inbetween': lr_inbetween,
                'lr_selector': lr_selector,
                'selector_budget_weight': selector_budget_weight,
                'selector_entropy_weight': selector_entropy_weight,
                'selector_aux_weight': cfg.selector_aux_weight,
                'selector_ratio': selector_ratio_avg,
                'selector_budget': selector_budget_avg,
                'selector_entropy': selector_entropy_avg,
                'selector_aux': selector_aux_avg,
            })

        if step % 2_000 == 0:
            _save_convergence_plot(metrics_logger)
        
        if step % 10_000 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'inbetween': inbetween_model.state_dict(),
                'inbetween_ema': ema_inbetween_state,
                'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
                'selector_ema': ema_selector_state if selector_has_trainable_params else None,
                'optimizer': inbetween_optimizer.state_dict(),
                'step': step,
                'best_step': best_step,
                'best_val_loss': best_val_loss,
                'cfg': cfg.__dict__,
            }, f'checkpoints/{ckpt_prefix}_step{step}.pt')
            print('Saved in-betweening checkpoint')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'inbetween': inbetween_model.state_dict(),
        'inbetween_ema': ema_inbetween_state,
        'selector': keyframe_selector.state_dict() if keyframe_selector is not None else None,
        'selector_ema': ema_selector_state if selector_has_trainable_params else None,
        'optimizer': inbetween_optimizer.state_dict(),
        'step': cfg.inbetween_steps,
        'best_step': best_step,
        'best_val_loss': best_val_loss,
        'cfg': cfg.__dict__,
    }, inbetween_final_ckpt_path)
    print(f'Saved final in-betweening checkpoint: {inbetween_final_ckpt_path}')
    if best_step > 0:
        print(f'Best checkpoint: {inbetween_best_ckpt_path} (step={best_step}, val_loss={best_val_loss:.6f})')
    _save_convergence_plot(metrics_logger)
    print(f"In-between convergence plot: {metrics_logger['plot_path']}")
    
    print("Diffusion in-betweening training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train composite motion models')
    parser.add_argument(
        '--stage',
        choices=['all', 'vqvae', 'gpt', 'inbetween'],
        default='all',
        help='Training stage to run (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if final checkpoints exist'
    )
    parser.add_argument('--vqvae-steps', type=int, default=None, help='Override VQ-VAE training steps')
    parser.add_argument('--gpt-steps', type=int, default=None, help='Override GPT training steps')
    parser.add_argument('--inbetween-steps', type=int, default=None, help='Override in-betweening training steps')
    parser.add_argument('--inbetween-resume', type=str, default=None, help='Path to in-betweening checkpoint to resume/fine-tune from')
    parser.add_argument('--inbetween-ckpt-prefix', type=str, default=None, help='Prefix for in-betweening checkpoint filenames; defaults to selector-specific naming')
    parser.add_argument('--keyframe-source-dir', type=str, default=None, help='Directory of external conditioning motions (id.npy) used for keyframes')
    parser.add_argument('--disable-selector', action='store_true', help='Disable learned keyframe selector and use dataset keyframes directly')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader workers')
    parser.add_argument('--scheduler-type', choices=['cosine', 'constant'], default=None, help='Learning rate scheduler type')
    parser.add_argument('--warmup-ratio', type=float, default=None, help='Warmup ratio for LR schedule')
    parser.add_argument('--min-lr-ratio', type=float, default=None, help='Minimum LR ratio for cosine decay')
    parser.add_argument('--inbetween-lr', type=float, default=None, help='Override in-between model LR')
    parser.add_argument('--selector-lr', type=float, default=None, help='Override selector LR')
    parser.add_argument('--selector-mode', choices=SELECTOR_MODE_CHOICES, default=None, help='Keyframe selector architecture')
    parser.add_argument('--selector-oracle-ckpt', type=str, default=None, help='Frozen in-between checkpoint used to build information-gain oracle labels')
    parser.add_argument('--selector-oracle-timesteps', type=int, default=None, help='Number of diffusion timesteps to average per oracle reconstruction score')
    parser.add_argument('--selector-eval-root', type=str, default=None, help='T2M-GPT root used to load the retrieval evaluator for retrieval-gain labels')
    parser.add_argument('--selector-retrieval-negatives', type=int, default=None, help='Number of negative captions sampled per oracle target for retrieval-gain')
    parser.add_argument('--ema-decay', type=float, default=None, help='EMA decay for in-between model/selector')
    parser.add_argument('--selector-curriculum-fraction', type=float, default=None, help='Fraction of total steps to ramp selector regularization')
    parser.add_argument('--val-interval', type=int, default=None, help='Validation interval for in-between training')
    parser.add_argument('--val-batches', type=int, default=None, help='Number of validation batches per eval pass')
    parser.add_argument('--grad-accum-steps', type=int, default=None, help='Gradient accumulation steps')
    args = parser.parse_args()
    main(
        args.stage,
        args.force,
        vqvae_steps=args.vqvae_steps,
        gpt_steps=args.gpt_steps,
        inbetween_steps=args.inbetween_steps,
        inbetween_resume=args.inbetween_resume,
        inbetween_ckpt_prefix=args.inbetween_ckpt_prefix,
        keyframe_source_dir=args.keyframe_source_dir,
        disable_selector=args.disable_selector,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        inbetween_lr=args.inbetween_lr,
        selector_lr=args.selector_lr,
        selector_mode=args.selector_mode,
        selector_oracle_ckpt=args.selector_oracle_ckpt,
        selector_oracle_timesteps=args.selector_oracle_timesteps,
        selector_eval_root=args.selector_eval_root,
        selector_retrieval_negatives=args.selector_retrieval_negatives,
        ema_decay=args.ema_decay,
        selector_curriculum_fraction=args.selector_curriculum_fraction,
        val_interval=args.val_interval,
        val_batches=args.val_batches,
        grad_accum_steps=args.grad_accum_steps,
    )
