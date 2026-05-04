"""Evaluate composite in-betweening and native T2M-GPT on HumanML3D."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader

from arlm_generate import ARLMConfig, _load_arlm_models
from config import CompositeConfig
from run_full_sample import (
    _convert_arlm_motion_to_local_stats,
    _default_inbetween_ckpt,
    _extract_step_from_path,
    _generate_arlm_motion,
    _load_arlm_stats,
    _resolve_arlm_ckpts,
)
from utils import encode_text, load_inbetween_model, select_keyframe_indices
from visualize import recover_from_ric


FOOT_JOINTS = (7, 10, 8, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate in-betweening and T2M-GPT models on HumanML3D"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Comma-separated models: all, t2mgpt, gpt, composite, "
            "composite_heuristic, composite_text_alignment, "
            "composite_information_gain, composite_retrieval_gain"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help=(
            "Comma-separated metrics or 'all': fid, diversity, jerk, foot_skating, "
            "multimodality, multimodal_distance, matching_score, r_precision"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count. Use 0 if multiprocessing causes issues.",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pinned memory when using CUDA.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--humanml-root", type=str, default="humanml")
    parser.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT")
    parser.add_argument(
        "--r-precision-top-k",
        type=str,
        default="1,2,3,5",
        help="Comma-separated cumulative R-precision cutoffs",
    )
    parser.add_argument(
        "--multimodal-repeats",
        type=int,
        default=10,
        help="Number of repeated samples per prompt for multimodal metrics",
    )
    parser.add_argument(
        "--multimodal-sample-count",
        type=int,
        default=10,
        help="Number of pair samples for standard multimodality metric",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="samples/eval_results",
        help="Directory for saved evaluation JSON",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Explicit JSON output path; overrides --results-dir naming",
    )
    parser.add_argument(
        "--load-results",
        action="store_true",
        help="Load existing JSON from --results-path instead of running evaluation",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Persist results JSON after evaluation",
    )
    parser.add_argument(
        "--categorical-sample",
        action="store_true",
        help="Use stochastic T2M-GPT categorical sampling for the first evaluation pass too",
    )
    parser.add_argument(
        "--ar-clamp-sigma",
        type=float,
        default=0.0,
        help="Clamp native T2M-GPT VQ outputs to +/- N sigma before denorm; <=0 disables",
    )
    parser.add_argument(
        "--composite-inbetween-ckpt",
        type=str,
        default=None,
        help="Explicit checkpoint override for all composite selector variants",
    )
    parser.add_argument(
        "--composite-text-alignment-ckpt",
        type=str,
        default=None,
        help="Checkpoint override for composite_text_alignment",
    )
    parser.add_argument(
        "--composite-information-gain-ckpt",
        type=str,
        default=None,
        help="Checkpoint override for composite_information_gain",
    )
    parser.add_argument(
        "--composite-retrieval-gain-ckpt",
        type=str,
        default=None,
        help="Checkpoint override for composite_retrieval_gain",
    )
    parser.add_argument(
        "--composite-inbetween-steps",
        type=int,
        default=None,
        help="Composite in-between step used for config metadata when ckpt step is not inferable",
    )
    parser.add_argument(
        "--composite-gpt-steps",
        type=int,
        default=None,
        help="Legacy local composite GPT step used only for config construction",
    )
    parser.add_argument(
        "--arlm-vq-ckpt",
        type=str,
        default=None,
        help="Path to T2M-GPT VQ checkpoint",
    )
    parser.add_argument(
        "--arlm-gpt-ckpt",
        type=str,
        default=None,
        help="Path to T2M-GPT GPT checkpoint",
    )
    parser.add_argument(
        "--disable-selector",
        action="store_true",
        help="Disable learned keyframe selector in composite diffusion evaluation",
    )
    parser.add_argument(
        "--keyframe-strategy",
        type=str,
        choices=["interval", "random"],
        default=None,
    )
    parser.add_argument("--keyframe-interval", type=int, default=5)
    parser.add_argument("--keyframe-count", type=int, default=None)
    parser.add_argument("--keyframe-min", type=int, default=None)
    parser.add_argument("--keyframe-max", type=int, default=None)
    parser.add_argument(
        "--no-keyframe-ends",
        action="store_true",
        help="Do not force the first/last keyframe in heuristic keyframe selection",
    )
    parser.add_argument(
        "--diff-guidance",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale for diffusion in-betweening",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_models(value: str) -> list[str]:
    composite_variants = [
        "composite_heuristic",
        "composite_text_alignment",
        "composite_information_gain",
        "composite_retrieval_gain",
    ]
    requested = []
    for model_name in parse_csv(value.lower()):
        if model_name == "all":
            requested.extend(["t2mgpt", *composite_variants])
        elif model_name == "composite":
            requested.extend(composite_variants)
        elif model_name == "gpt":
            requested.append("t2mgpt")
        elif model_name in {"t2mgpt", *composite_variants}:
            requested.append(model_name)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    if not requested:
        raise ValueError("No models selected for evaluation")
    return list(dict.fromkeys(requested))


def resolve_metrics(value: str) -> set[str]:
    all_metrics = {
        "fid",
        "diversity",
        "jerk",
        "foot_skating",
        "multimodality",
        "multimodal_distance",
        "matching_score",
        "r_precision",
    }
    metrics = set(parse_csv(value.lower()))
    if not metrics or "all" in metrics:
        return all_metrics
    unknown = metrics - all_metrics
    if unknown:
        raise ValueError(f"Unsupported metric names: {sorted(unknown)}")
    return metrics


def parse_top_k(value: str) -> list[int]:
    top_k = sorted({int(item) for item in parse_csv(value)})
    if not top_k or top_k[0] < 1:
        raise ValueError("R-precision top-k must contain positive integers")
    return top_k


def ensure_results_path(args: argparse.Namespace) -> str:
    if args.results_path:
        return os.path.abspath(args.results_path)
    os.makedirs(args.results_dir, exist_ok=True)
    model_tag = "-".join(resolve_models(args.models))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(os.path.join(args.results_dir, f"eval_{model_tag}_{stamp}.json"))


@contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def prepare_t2mgpt_imports(t2mgpt_root: str) -> None:
    if t2mgpt_root not in sys.path:
        sys.path.insert(0, t2mgpt_root)

    for package_name in ("dataset", "models", "options", "utils"):
        existing = sys.modules.get(package_name)
        if existing is not None and hasattr(existing, "__file__"):
            del sys.modules[package_name]

    package_paths = {
        "dataset": os.path.join(t2mgpt_root, "dataset"),
        "models": os.path.join(t2mgpt_root, "models"),
        "options": os.path.join(t2mgpt_root, "options"),
        "utils": os.path.join(t2mgpt_root, "utils"),
    }
    for package_name, package_path in package_paths.items():
        module = types.ModuleType(package_name)
        module.__path__ = [package_path]  # type: ignore[attr-defined]
        sys.modules[package_name] = module


def build_eval_loader_and_wrapper(
    t2mgpt_root: str,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    prepare_t2mgpt_imports(t2mgpt_root)
    with pushd(t2mgpt_root):
        from dataset.dataset_TM_eval import Text2MotionDataset, collate_fn  # type: ignore
        from models.evaluator_wrapper import EvaluatorModelWrapper  # type: ignore
        from options.get_eval_option import get_opt  # type: ignore
        from utils.word_vectorizer import WordVectorizer  # type: ignore

        word_vectorizer = WordVectorizer("./glove", "our_vab")
        dataset = Text2MotionDataset("t2m", "test", word_vectorizer)
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory and device.type == "cuda"),
            "collate_fn": collate_fn,
            "drop_last": False,
        }
        if int(num_workers) > 0:
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(dataset, **loader_kwargs)
        dataset_opt_path = os.path.join(
            t2mgpt_root,
            "checkpoints",
            "t2m",
            "Comp_v6_KLD005",
            "opt.txt",
        )
        wrapper_opt = get_opt(dataset_opt_path, device)
        wrapper_opt.checkpoints_dir = os.path.join(t2mgpt_root, "checkpoints")
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    return loader, eval_wrapper


def truncate_batch(batch: tuple[Any, ...], limit: int) -> tuple[Any, ...]:
    if limit <= 0:
        raise ValueError("Batch truncation limit must be positive")
    truncated: list[Any] = []
    for item in batch:
        if torch.is_tensor(item):
            truncated.append(item[:limit])
        elif isinstance(item, np.ndarray):
            truncated.append(item[:limit])
        elif isinstance(item, (list, tuple)):
            truncated.append(item[:limit])
        else:
            truncated.append(item)
    return tuple(truncated)


def normalize_motion(raw_motion: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (raw_motion - mean) / (std + 1e-8)


def pad_motion(normalized_motion: np.ndarray, max_len: int = 196) -> tuple[np.ndarray, int]:
    seq_len = int(min(len(normalized_motion), max_len))
    feature_dim = int(normalized_motion.shape[1])
    padded = np.zeros((max_len, feature_dim), dtype=np.float32)
    if seq_len > 0:
        padded[:seq_len] = normalized_motion[:seq_len]
    return padded, seq_len


def joints_from_motion(raw_motion: np.ndarray) -> np.ndarray:
    return recover_from_ric(raw_motion.astype(np.float32), joints_num=22)


def compute_jerk(joints: np.ndarray) -> float:
    if joints.shape[0] < 4:
        return 0.0
    velocity = np.diff(joints, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    return float(jerk_mag.mean())


def compute_foot_skating(joints: np.ndarray, height_thresh: float = 0.05) -> float:
    if joints.shape[0] < 2:
        return 0.0
    foot = joints[:, FOOT_JOINTS, :]
    floor_height = float(np.min(foot[..., 1]))
    heights = foot[:-1, :, 1] - floor_height
    contact = heights < height_thresh
    horizontal_velocity = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
    if not np.any(contact):
        return float(horizontal_velocity.mean())
    return float(horizontal_velocity[contact].mean())


def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(np.maximum(d1 + d2 + d3, 0.0))
    return dists


def calculate_top_k(arg_sorted: np.ndarray, top_k: int) -> np.ndarray:
    size = arg_sorted.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = arg_sorted == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    return np.concatenate(top_k_list, axis=1)


def calculate_r_precision(
    text_embeddings: np.ndarray,
    motion_embeddings: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, float]:
    dist_mat = euclidean_distance_matrix(text_embeddings, motion_embeddings)
    matching_score = float(np.trace(dist_mat))
    arg_sorted = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(arg_sorted, max_k)
    return top_k_mat.sum(axis=0), matching_score


def calculate_diversity(activation: np.ndarray, diversity_times: int) -> float:
    if activation.shape[0] <= 1:
        return 0.0
    diversity_times = min(diversity_times, activation.shape[0] - 1)
    first_indices = np.random.choice(activation.shape[0], diversity_times, replace=False)
    second_indices = np.random.choice(activation.shape[0], diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return float(dist.mean())


def calculate_multimodality(activation: np.ndarray, sample_count: int) -> float:
    if activation.ndim != 3 or activation.shape[1] <= 1:
        return 0.0
    sample_count = min(sample_count, activation.shape[1] - 1)
    first_indices = np.random.choice(activation.shape[1], sample_count, replace=False)
    second_indices = np.random.choice(activation.shape[1], sample_count, replace=False)
    dist = linalg.norm(activation[:, first_indices] - activation[:, second_indices], axis=2)
    return float(dist.mean())


def calculate_activation_statistics(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_multimodal_distance(multimodal_embeddings: np.ndarray) -> float:
    if multimodal_embeddings.ndim != 3 or multimodal_embeddings.shape[1] <= 1:
        return 0.0
    distances = []
    for row in multimodal_embeddings:
        diff = row[:, None, :] - row[None, :, :]
        pairwise = np.linalg.norm(diff, axis=-1)
        tri = pairwise[np.triu_indices(pairwise.shape[0], k=1)]
        if tri.size > 0:
            distances.append(float(tri.mean()))
    if not distances:
        return 0.0
    return float(np.mean(distances))


@dataclass
class MotionBatch:
    padded: torch.Tensor
    lengths: torch.Tensor
    raw_motions: list[np.ndarray]
    jerk_scores: list[float]
    foot_skating_scores: list[float]


class BaseGenerator:
    name: str

    def generate(self, prompt: str, target_length: int, stochastic: bool) -> np.ndarray:
        raise NotImplementedError


class T2MGPTGenerator(BaseGenerator):
    def __init__(
        self,
        t2mgpt_root: str,
        device: torch.device,
        arlm_vq_ckpt: str | None,
        arlm_gpt_ckpt: str | None,
        clamp_sigma: float,
    ):
        self.name = "t2mgpt"
        self.device = device
        self.t2mgpt_root = os.path.abspath(t2mgpt_root)
        self.arlm_cfg = ARLMConfig()
        vq_ckpt, gpt_ckpt = _resolve_arlm_ckpts(self.t2mgpt_root, arlm_vq_ckpt, arlm_gpt_ckpt)
        self.clip_lib, self.clip_model, self.vq_model, self.gpt_model = _load_arlm_models(
            self.t2mgpt_root,
            self.arlm_cfg,
            vq_ckpt,
            gpt_ckpt,
            device,
        )
        self.eval_mean, self.eval_std, _ = _load_arlm_stats(self.t2mgpt_root, device)
        self._text_cache: dict[str, torch.Tensor] = {}
        self._cond_uncond: torch.Tensor | None = None

    def _encode_prompt_cached(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self._text_cache.get(prompt)
        if cond is None:
            cond = encode_text(self.clip_model, [prompt])
            self._text_cache[prompt] = cond
        if self._cond_uncond is None:
            self._cond_uncond = encode_text(self.clip_model, [""])
        return cond, self._cond_uncond
        self.clamp_sigma = float(clamp_sigma)

    def generate(self, prompt: str, target_length: int, stochastic: bool) -> np.ndarray:
        del target_length
        native_norm = _generate_arlm_motion(
            prompt=prompt,
            clip_lib=self.clip_lib,
            clip_model=self.clip_model,
            vq_model=self.vq_model,
            gpt_model=self.gpt_model,
            device=self.device,
            categorical_sample=bool(stochastic),
        )
        if self.clamp_sigma > 0:
            native_norm = native_norm.clamp(-self.clamp_sigma, self.clamp_sigma)
        raw_motion = native_norm * (self.eval_std + 1e-8) + self.eval_mean
        return raw_motion.detach().cpu().numpy().astype(np.float32)


class CompositeGenerator(BaseGenerator):
    def __init__(
        self,
        humanml_root: str,
        t2mgpt_root: str,
        device: torch.device,
        inbetween_ckpt: str | None,
        arlm_vq_ckpt: str | None,
        arlm_gpt_ckpt: str | None,
        selector_mode: str,
        disable_selector: bool,
        diff_guidance: float,
        clamp_sigma: float,
    ):
        self.name = "composite"
        self.device = device
        self.t2mgpt_root = os.path.abspath(t2mgpt_root)
        self.diff_guidance = float(diff_guidance)
        self.disable_selector = bool(disable_selector)
        self.clamp_sigma = float(clamp_sigma)

        if inbetween_ckpt is None:
            resolved_ckpt = _default_inbetween_ckpt_for_mode(selector_mode)
        else:
            resolved_ckpt = os.path.abspath(inbetween_ckpt)

        cfg = CompositeConfig(
            root=os.path.abspath(humanml_root),
            selector_mode=selector_mode,
        )
        if disable_selector:
            cfg.use_learned_keyframe_selector = False
        self.cfg = cfg
        self.inbetween_model, self.diff_inbetween, self.clip_model, local_mean, local_std, self.fdim = load_inbetween_model(
            cfg,
            str(device),
            inbetween_ckpt_path=resolved_ckpt,
        )
        self.local_mean = local_mean.to(device)
        self.local_std = local_std.to(device)

        self.arlm_cfg = ARLMConfig()
        vq_ckpt, gpt_ckpt = _resolve_arlm_ckpts(self.t2mgpt_root, arlm_vq_ckpt, arlm_gpt_ckpt)
        self.clip_lib_arlm, self.clip_model_arlm, self.vq_model_arlm, self.gpt_model_arlm = _load_arlm_models(
            self.t2mgpt_root,
            self.arlm_cfg,
            vq_ckpt,
            gpt_ckpt,
            device,
        )
        self.eval_mean, self.eval_std, _ = _load_arlm_stats(self.t2mgpt_root, device)

    def _select_keyframes(self, ar_motion_norm: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        effective_length = int(ar_motion_norm.shape[0])
        strategy = self.cfg.keyframe_strategy
        keyframe_count = self.cfg.keyframe_count
        keyframe_min = self.cfg.keyframe_min
        keyframe_max = self.cfg.keyframe_max
        selector_model = getattr(self.inbetween_model, "keyframe_selector", None)
        use_selector = self.cfg.use_learned_keyframe_selector and (not self.disable_selector)
        if use_selector and selector_model is not None:
            selector_model.eval()
            selector_valid = torch.ones(1, effective_length, dtype=torch.bool, device=self.device)
            _, selector_mask_st = selector_model(
                ar_motion_norm.unsqueeze(0),
                selector_valid,
                cond=cond,
            )
            keyframe_indices = torch.nonzero(selector_mask_st[0] > 0.5, as_tuple=False).squeeze(1)
            if keyframe_indices.numel() > 2:
                return keyframe_indices

        idx_list = select_keyframe_indices(
            length=effective_length,
            keyframe_interval=self.cfg.keyframe_interval,
            strategy=strategy,
            keyframe_count=keyframe_count,
            keyframe_min=keyframe_min,
            keyframe_max=keyframe_max,
            include_ends=self.cfg.keyframe_include_ends,
        )
        return torch.tensor(idx_list, dtype=torch.long, device=self.device)

    def generate(self, prompt: str, target_length: int, stochastic: bool) -> np.ndarray:
        del target_length
        native_norm = _generate_arlm_motion(
            prompt=prompt,
            clip_lib=self.clip_lib_arlm,
            clip_model=self.clip_model_arlm,
            vq_model=self.vq_model_arlm,
            gpt_model=self.gpt_model_arlm,
            device=self.device,
            categorical_sample=bool(stochastic),
        )
        _, ar_motion_norm = _convert_arlm_motion_to_local_stats(
            ar_motion_native_norm=native_norm,
            arlm_mean=self.eval_mean,
            arlm_std=self.eval_std,
            local_mean=self.local_mean,
            local_std=self.local_std,
            clamp_sigma=self.clamp_sigma,
        )

        cond, cond_uncond = self._encode_prompt_cached(prompt)
        keyframe_indices = self._select_keyframes(ar_motion_norm, cond)
        keyframes_norm = ar_motion_norm[keyframe_indices]

        time_steps = int(ar_motion_norm.shape[0])
        valid_mask = torch.ones(1, time_steps, dtype=torch.bool, device=self.device)
        keyframe_mask = torch.ones(1, keyframe_indices.numel(), dtype=torch.bool, device=self.device)
        sampled_norm = self.diff_inbetween.sample_inbetween(
            model=self.inbetween_model,
            shape=(1, time_steps, self.fdim),
            cond=cond,
            mask=valid_mask,
            keyframes=keyframes_norm.unsqueeze(0),
            keyframe_indices=keyframe_indices.unsqueeze(0),
            keyframe_mask=keyframe_mask,
            guidance_scale=self.diff_guidance,
            cond_uncond=cond_uncond,
            source_motion=ar_motion_norm.unsqueeze(0),
            text_prompt=prompt,
        )[0]
        raw_motion = sampled_norm * (self.local_std + 1e-8) + self.local_mean
        return raw_motion.detach().cpu().numpy().astype(np.float32)


def build_motion_batch(
    generator: BaseGenerator,
    captions: Iterable[str],
    lengths: Iterable[int],
    eval_mean: np.ndarray,
    eval_std: np.ndarray,
    stochastic: bool,
    need_jerk: bool = False,
    need_foot_skating: bool = False,
) -> MotionBatch:
    raw_motions: list[np.ndarray] = []
    jerk_scores: list[float] = []
    foot_scores: list[float] = []
    padded = []
    padded_lengths = []
    for caption, length in zip(captions, lengths):
        raw_motion = generator.generate(caption, int(length), stochastic=stochastic)
        raw_motion = raw_motion.astype(np.float32)
        raw_motions.append(raw_motion)
        if need_jerk or need_foot_skating:
            joints = joints_from_motion(raw_motion)
            if need_jerk:
                jerk_scores.append(compute_jerk(joints))
            if need_foot_skating:
                foot_scores.append(compute_foot_skating(joints))
        normalized = normalize_motion(raw_motion, eval_mean, eval_std)
        padded_motion, seq_len = pad_motion(normalized)
        padded.append(torch.from_numpy(padded_motion))
        padded_lengths.append(seq_len)
    return MotionBatch(
        padded=torch.stack(padded, dim=0),
        lengths=torch.tensor(padded_lengths, dtype=torch.long),
        raw_motions=raw_motions,
        jerk_scores=jerk_scores,
        foot_skating_scores=foot_scores,
    )


def evaluate_model(
    model_name: str,
    generator: BaseGenerator,
    loader: DataLoader,
    eval_wrapper,
    top_k: list[int],
    metrics: set[str],
    num_samples: int,
    multimodal_repeats: int,
    multimodal_sample_count: int,
    categorical_sample: bool,
) -> dict[str, Any]:
    device = next(eval_wrapper.motion_encoder.parameters()).device
    eval_mean = loader.dataset.mean.astype(np.float32)
    eval_std = loader.dataset.std.astype(np.float32)

    max_k = max(top_k)
    processed = 0
    r_precision_sums = np.zeros(max_k, dtype=np.float64)
    matching_score_total = 0.0
    gt_motion_embeddings = []
    pred_motion_embeddings = []
    jerk_scores: list[float] = []
    foot_scores: list[float] = []
    multimodal_embeddings = []

    for batch in loader:
        remaining = num_samples - processed
        if remaining <= 0:
            break
        word_embeddings, pos_one_hots, captions, sent_len, motion, m_length, token, name = batch
        batch_size = len(captions)
        if batch_size > remaining:
            batch = truncate_batch(batch, remaining)
            word_embeddings, pos_one_hots, captions, sent_len, motion, m_length, token, name = batch
            batch_size = remaining

        motion = motion.to(device).float()
        m_length = m_length.to(device).long()

        text_embeddings, gt_embeddings = eval_wrapper.get_co_embeddings(
            word_embeddings,
            pos_one_hots,
            sent_len,
            motion,
            m_length,
        )

        first_pass = build_motion_batch(
            generator=generator,
            captions=captions,
            lengths=m_length.detach().cpu().tolist(),
            eval_mean=eval_mean,
            eval_std=eval_std,
            stochastic=categorical_sample,
            need_jerk="jerk" in metrics,
            need_foot_skating="foot_skating" in metrics,
        )

        pred_padded = first_pass.padded.to(device)
        pred_lengths = first_pass.lengths.to(device)
        _, pred_embeddings = eval_wrapper.get_co_embeddings(
            word_embeddings,
            pos_one_hots,
            sent_len,
            pred_padded,
            pred_lengths,
        )

        if "r_precision" in metrics or "matching_score" in metrics:
            temp_r, temp_match = calculate_r_precision(
                text_embeddings.detach().cpu().numpy(),
                pred_embeddings.detach().cpu().numpy(),
                max_k=max_k,
            )
            r_precision_sums += temp_r
            matching_score_total += temp_match

        gt_motion_embeddings.append(gt_embeddings.detach().cpu().numpy())
        pred_motion_embeddings.append(pred_embeddings.detach().cpu().numpy())
        if "jerk" in metrics:
            jerk_scores.extend(first_pass.jerk_scores)
        if "foot_skating" in metrics:
            foot_scores.extend(first_pass.foot_skating_scores)

        if "multimodality" in metrics or "multimodal_distance" in metrics:
            repeated_embeddings = [pred_embeddings.detach().cpu().numpy()]
            for _ in range(max(0, multimodal_repeats - 1)):
                extra_pass = build_motion_batch(
                    generator=generator,
                    captions=captions,
                    lengths=m_length.detach().cpu().tolist(),
                    eval_mean=eval_mean,
                    eval_std=eval_std,
                    stochastic=True,
                    need_jerk=False,
                    need_foot_skating=False,
                )
                extra_padded = extra_pass.padded.to(device)
                extra_lengths = extra_pass.lengths.to(device)
                _, extra_embeddings = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    extra_padded,
                    extra_lengths,
                )
                repeated_embeddings.append(extra_embeddings.detach().cpu().numpy())
            multimodal_embeddings.append(np.stack(repeated_embeddings, axis=1))

        processed += batch_size
        print(f"[{model_name}] processed {processed}/{num_samples} samples")

    gt_motion_np = np.concatenate(gt_motion_embeddings, axis=0)
    pred_motion_np = np.concatenate(pred_motion_embeddings, axis=0)

    results: dict[str, Any] = {
        "model": model_name,
        "num_samples": int(processed),
    }

    if "fid" in metrics:
        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_np)
        pred_mu, pred_cov = calculate_activation_statistics(pred_motion_np)
        results["fid"] = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)

    if "diversity" in metrics:
        diversity_times = 300 if pred_motion_np.shape[0] > 300 else max(1, pred_motion_np.shape[0] - 1)
        results["diversity"] = calculate_diversity(pred_motion_np, diversity_times)

    if "matching_score" in metrics:
        results["matching_score"] = float(matching_score_total / max(processed, 1))

    if "r_precision" in metrics:
        r_precision = {}
        for k in top_k:
            r_precision[f"R@{k}"] = float(r_precision_sums[k - 1] / max(processed, 1))
        results["r_precision"] = r_precision

    if "jerk" in metrics:
        results["jerk"] = float(np.mean(jerk_scores)) if jerk_scores else 0.0

    if "foot_skating" in metrics:
        results["foot_skating"] = float(np.mean(foot_scores)) if foot_scores else 0.0

    if "multimodality" in metrics or "multimodal_distance" in metrics:
        multimodal_np = np.concatenate(multimodal_embeddings, axis=0) if multimodal_embeddings else np.zeros((0, 0, 0), dtype=np.float32)
        if "multimodality" in metrics:
            results["multimodality"] = calculate_multimodality(multimodal_np, multimodal_sample_count)
        if "multimodal_distance" in metrics:
            results["multimodal_distance"] = calculate_multimodal_distance(multimodal_np)

    return results


def print_results(all_results: dict[str, Any]) -> None:
    print("\nEvaluation summary")
    print(json.dumps(all_results, indent=2, sort_keys=True))


def _default_inbetween_ckpt_for_mode(selector_mode: str) -> str:
    mode = str(selector_mode).strip().lower().replace("-", "_")
    if mode == "heuristic":
        return _default_inbetween_ckpt()

    preferred = [
        os.path.join("checkpoints", f"finetuned_inbetween_{mode}_best.pt"),
        os.path.join("checkpoints", f"composite_inbetween_{mode}_best.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    step_candidates = glob.glob(os.path.join("checkpoints", f"composite_inbetween_{mode}_step*.pt"))
    if step_candidates:
        def _extract_step(path: str) -> int:
            step = _extract_step_from_path(path)
            return -1 if step is None else int(step)

        return max(step_candidates, key=_extract_step)

    fallback = _default_inbetween_ckpt()
    print(f"Warning: no mode-specific checkpoint found for selector mode '{mode}', falling back to {fallback}")
    return fallback


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t2mgpt_root = os.path.abspath(args.t2mgpt_root)
    results_path = ensure_results_path(args)

    if args.load_results:
        with open(results_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        print_results(loaded)
        return

    models = resolve_models(args.models)
    metrics = resolve_metrics(args.metrics)
    top_k = parse_top_k(args.r_precision_top_k)

    loader, eval_wrapper = build_eval_loader_and_wrapper(
        t2mgpt_root=t2mgpt_root,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    generators: dict[str, BaseGenerator] = {}
    if "t2mgpt" in models:
        generators["t2mgpt"] = T2MGPTGenerator(
            t2mgpt_root=t2mgpt_root,
            device=device,
            arlm_vq_ckpt=args.arlm_vq_ckpt,
            arlm_gpt_ckpt=args.arlm_gpt_ckpt,
            clamp_sigma=args.ar_clamp_sigma,
        )
    composite_specs = {
        "composite_heuristic": {
            "selector_mode": "heuristic",
            "disable_selector": True,
            "ckpt_override": None,
        },
        "composite_text_alignment": {
            "selector_mode": "text_alignment",
            "disable_selector": False,
            "ckpt_override": args.composite_text_alignment_ckpt,
        },
        "composite_information_gain": {
            "selector_mode": "information_gain",
            "disable_selector": False,
            "ckpt_override": args.composite_information_gain_ckpt,
        },
        "composite_retrieval_gain": {
            "selector_mode": "retrieval_gain",
            "disable_selector": False,
            "ckpt_override": args.composite_retrieval_gain_ckpt,
        },
    }
    for model_name, spec in composite_specs.items():
        if model_name not in models:
            continue

        generators[model_name] = CompositeGenerator(
            humanml_root=args.humanml_root,
            t2mgpt_root=t2mgpt_root,
            device=device,
            inbetween_ckpt=args.composite_inbetween_ckpt or spec["ckpt_override"],
            arlm_vq_ckpt=args.arlm_vq_ckpt,
            arlm_gpt_ckpt=args.arlm_gpt_ckpt,
            selector_mode=spec["selector_mode"],
            disable_selector=spec["disable_selector"] or args.disable_selector,
            diff_guidance=args.diff_guidance,
            clamp_sigma=args.ar_clamp_sigma,
        )

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "metrics": sorted(metrics),
        "models": {},
        "num_samples": int(args.num_samples),
        "multimodal_repeats": int(args.multimodal_repeats),
    }

    for model_name in models:
        print(f"Running evaluation for {model_name} on {device}")
        with torch.inference_mode():
            all_results["models"][model_name] = evaluate_model(
                model_name=model_name,
                generator=generators[model_name],
                loader=loader,
                eval_wrapper=eval_wrapper,
                top_k=top_k,
                metrics=metrics,
                num_samples=args.num_samples,
                multimodal_repeats=args.multimodal_repeats,
                multimodal_sample_count=args.multimodal_sample_count,
                categorical_sample=args.categorical_sample,
            )

    if args.save_results:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved results to {results_path}")

    print_results(all_results)


if __name__ == "__main__":
    main()