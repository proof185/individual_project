"""Evaluate keyframe selection strategies against native T2M-GPT on HumanML3D."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CompositeConfig, EvalConfig
from condmdi_adapter import load_external_condmdi_runtime
from keyframe_selectors import SELECTOR_MODE_CHOICES, build_keyframe_selector
from utils import encode_text, load_inbetween_model, select_keyframe_indices


FOOT_JOINTS = (7, 10, 8, 11)
COMPOSITE_SELECTOR_MODES = tuple(SELECTOR_MODE_CHOICES)
COMPOSITE_MODEL_ALIASES = {
    "composite_heuristic": "composite_random",
}
SELECTION_STRATEGY_ALIASES = {
    mode: f"composite_{mode}" for mode in COMPOSITE_SELECTOR_MODES
}
CONDMDI_UNCONDITIONAL_MODEL = "condmdi_unconditional"
CONDMDI_UNCONDITIONAL_ALIASES = {
    "condmdi_uncond": CONDMDI_UNCONDITIONAL_MODEL,
    "condmdi_text_only": CONDMDI_UNCONDITIONAL_MODEL,
    "condmdi": CONDMDI_UNCONDITIONAL_MODEL,
}
T2MGPT_ALIASES = {
    "gpt": "t2mgpt",
    "t2m-gpt": "t2mgpt",
    "t2m_gpt": "t2mgpt",
}
_INBETWEEN_RUNTIME_CACHE: dict[
    tuple[str, str, str],
    tuple[object, object, object, torch.Tensor, torch.Tensor, int],
] = {}
_CONDMDI_TEXT_RUNTIME_CACHE: dict[
    tuple[str, str, str],
    tuple[object, object, torch.Tensor, torch.Tensor, int],
] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the configured HumanML3D evaluation and write a CSV summary."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override EvalConfig.seed")
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
    composite_variants = [f"composite_{mode}" for mode in COMPOSITE_SELECTOR_MODES]
    strategy_names = ", ".join(COMPOSITE_SELECTOR_MODES)
    requested = []
    for model_name in parse_csv(value.lower()):
        model_name = COMPOSITE_MODEL_ALIASES.get(model_name, model_name)
        model_name = CONDMDI_UNCONDITIONAL_ALIASES.get(model_name, model_name)
        model_name = T2MGPT_ALIASES.get(model_name, model_name)
        model_name = SELECTION_STRATEGY_ALIASES.get(model_name, model_name)
        if model_name == "all":
            requested.extend(["t2mgpt", CONDMDI_UNCONDITIONAL_MODEL, *composite_variants])
        elif model_name == "composite":
            requested.extend(composite_variants)
        elif model_name in {"t2mgpt", CONDMDI_UNCONDITIONAL_MODEL, *composite_variants}:
            requested.append(model_name)
        else:
            raise ValueError(
                f"Unsupported model name: {model_name}. "
                f"Use t2mgpt, {CONDMDI_UNCONDITIONAL_MODEL}, all, or one of these selection strategies: {strategy_names}."
            )
    if not requested:
        raise ValueError("No models selected for evaluation")
    return list(dict.fromkeys(requested))


def resolve_metrics(value: str) -> set[str]:
    all_metrics = {
        "fid",
        "diversity",
        "jerk",
        "foot_skating",
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


def ensure_results_path(cfg: EvalConfig) -> str:
    if cfg.results_path:
        return os.path.abspath(cfg.results_path)
    os.makedirs(cfg.results_dir, exist_ok=True)
    resolved = resolve_models(cfg.models)
    model_tag = "-".join(resolved) if len("-".join(resolved)) <= 50 else f"{len(resolved)}models"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(os.path.join(cfg.results_dir, f"eval_{model_tag}_seed{cfg.seed}_{stamp}.json"))


def ensure_csv_path(cfg: EvalConfig, results_path: str) -> str:
    if cfg.csv_path:
        return os.path.abspath(cfg.csv_path)
    base, _ = os.path.splitext(results_path)
    return f"{base}.csv"


def _latest_step(checkpoint_glob: str, prefix: str) -> int:
    paths = glob.glob(checkpoint_glob)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {checkpoint_glob}")

    best = -1
    for path in paths:
        base = os.path.basename(path)
        if not base.startswith(prefix):
            continue
        step_part = base[len(prefix):].removesuffix(".pt")
        if step_part.isdigit():
            best = max(best, int(step_part))

    if best < 0:
        raise RuntimeError(f"Could not parse checkpoint step from files matching: {checkpoint_glob}")
    return best


def _extract_step_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def _project_path(*parts: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *parts))


def _default_inbetween_ckpt() -> str:
    preferred = [
        _project_path("checkpoints", "finetuned_inbetween_best.pt"),
        _project_path("checkpoints", "composite_inbetween_best.pt"),
        _project_path("..", "diffusion-motion-inbetweening", "save", "condmdi_randomframes", "model000750000.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    step = _latest_step(
        checkpoint_glob=_project_path("checkpoints", "composite_inbetween_step*.pt"),
        prefix="composite_inbetween_step",
    )
    return _project_path("checkpoints", f"composite_inbetween_step{step}.pt")


def _resolve_arlm_ckpts(
    t2mgpt_root: str,
    vq_ckpt: str | None,
    gpt_ckpt: str | None,
) -> tuple[str, str]:
    if not vq_ckpt:
        vq_ckpt = os.path.join(t2mgpt_root, "pretrained", "VQVAE", "net_last.pth")
    if not gpt_ckpt:
        gpt_ckpt = os.path.join(t2mgpt_root, "pretrained", "VQTransformer_corruption05", "net_best_fid.pth")
    return os.path.abspath(vq_ckpt), os.path.abspath(gpt_ckpt)


def _load_arlm_stats(t2mgpt_root: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
    meta_dir = os.path.join(t2mgpt_root, "checkpoints", "t2m", "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    mean_path = os.path.join(meta_dir, "mean.npy")
    std_path = os.path.join(meta_dir, "std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            "Missing native T2M-GPT normalization stats. Expected:\n"
            f"  {mean_path}\n"
            f"  {std_path}"
        )
    mean = torch.from_numpy(np.load(mean_path)).float().to(device)
    std = torch.from_numpy(np.load(std_path)).float().to(device)
    return mean, std, meta_dir


def _load_arlm_runtime(t2mgpt_root: str):
    try:
        from arlm_generate import ARLMConfig, _load_arlm_models  # type: ignore
        return ARLMConfig, _load_arlm_models
    except ModuleNotFoundError:
        pass

    prepare_t2mgpt_imports(t2mgpt_root)
    with pushd(t2mgpt_root):
        import clip as clip_lib  # type: ignore
        from models.t2m_trans import Text2Motion_Transformer  # type: ignore
        from models.vqvae import HumanVQVAE  # type: ignore

    @dataclass
    class ARLMConfig:
        pass

    def _load_arlm_models(
        root: str,
        cfg: ARLMConfig,
        vq_ckpt: str,
        gpt_ckpt: str,
        device: torch.device,
    ):
        del cfg
        prepare_t2mgpt_imports(root)
        with pushd(root):
            model_cfg = SimpleNamespace(dataname="t2m", quantizer="ema_reset", mu=0.99)
            vq_model = HumanVQVAE(
                model_cfg,
                nb_code=512,
                code_dim=512,
                output_emb_width=512,
                down_t=2,
                stride_t=2,
                width=512,
                depth=3,
                dilation_growth_rate=3,
            ).to(device)
            gpt_model = Text2Motion_Transformer(
                num_vq=512,
                embed_dim=1024,
                clip_dim=512,
                block_size=51,
                num_layers=9,
                n_head=16,
                drop_out_rate=0.1,
                fc_rate=4,
            ).to(device)

            vq_state = torch.load(vq_ckpt, map_location="cpu")
            gpt_state = torch.load(gpt_ckpt, map_location="cpu")
            vq_model.load_state_dict(vq_state["net"], strict=True)
            gpt_model.load_state_dict(gpt_state["trans"], strict=True)
            vq_model.eval()
            gpt_model.eval()

            clip_model, _ = clip_lib.load("ViT-B/32", device=device, jit=False)
            clip_lib.model.convert_weights(clip_model)
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad = False
        return clip_lib, clip_model, vq_model, gpt_model

    return ARLMConfig, _load_arlm_models


def _generate_arlm_motion(
    prompt: str,
    clip_lib,
    clip_model,
    vq_model,
    gpt_model,
    device: torch.device,
    categorical_sample: bool,
) -> torch.Tensor:
    with torch.no_grad():
        text_tokens = clip_lib.tokenize([prompt], truncate=True).to(device)
        clip_feat = clip_model.encode_text(text_tokens).float()
        token_idx = None
        sample_modes = [bool(categorical_sample)]
        if categorical_sample:
            sample_modes.extend([True, True, False])

        for sample_mode in sample_modes:
            candidate = gpt_model.sample(clip_feat, if_categorial=sample_mode)
            if candidate is not None and candidate.ndim == 2 and candidate.shape[1] > 0:
                token_idx = candidate
                break

        if token_idx is None:
            raise RuntimeError(
                "T2M-GPT sampling produced an empty token sequence even after fallback to greedy decoding."
            )
        return vq_model.forward_decoder(token_idx)[0]


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
    num_workers: int = 0,
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
    from visualize import recover_from_ric

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
        if i < bool_mat.shape[1]:
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


@dataclass
class MotionBatch:
    padded: torch.Tensor
    lengths: torch.Tensor
    raw_motions: list[np.ndarray]
    jerk_scores: list[float]
    foot_skating_scores: list[float]


@dataclass
class GeneratedARMotion:
    raw_motion: np.ndarray


@dataclass
class CompositeConditioning:
    prompt: str
    ar_motion_norm: torch.Tensor
    cond: torch.Tensor
    cond_uncond: torch.Tensor
    keyframe_indices: torch.Tensor


class T2MGPTMotionSource:
    def __init__(
        self,
        t2mgpt_root: str,
        device: torch.device,
        arlm_vq_ckpt: str | None,
        arlm_gpt_ckpt: str | None,
        clamp_sigma: float,
    ):
        self.device = device
        self.t2mgpt_root = os.path.abspath(t2mgpt_root)
        ARLMConfig, load_arlm_models = _load_arlm_runtime(self.t2mgpt_root)
        self.arlm_cfg = ARLMConfig()
        vq_ckpt, gpt_ckpt = _resolve_arlm_ckpts(self.t2mgpt_root, arlm_vq_ckpt, arlm_gpt_ckpt)
        self.clip_lib, self.clip_model, self.vq_model, self.gpt_model = load_arlm_models(
            self.t2mgpt_root,
            self.arlm_cfg,
            vq_ckpt,
            gpt_ckpt,
            device,
        )
        self.eval_mean, self.eval_std, _ = _load_arlm_stats(self.t2mgpt_root, device)
        self.clamp_sigma = float(clamp_sigma)

    def generate(self, prompt: str, stochastic: bool) -> GeneratedARMotion:
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
        return GeneratedARMotion(raw_motion=raw_motion.detach().cpu().numpy().astype(np.float32))


class CompositeStrategy:
    def __init__(
        self,
        name: str,
        humanml_root: str,
        device: torch.device,
        inbetween_ckpt: str | None,
        selector_mode: str,
        disable_selector: bool,
        diff_guidance: float,
        fallback_keyframe_strategy: str | None,
        keyframe_interval: int,
        keyframe_count: int | None,
        keyframe_min: int | None,
        keyframe_max: int | None,
        include_ends: bool,
        keyframe_topk: int | None,
        keyframe_budget_ratio: float,
        ddim_steps: int = 50,
    ):
        self.name = name
        self.device = device
        self.selector_mode = selector_mode
        self.diff_guidance = float(diff_guidance)
        self.disable_selector = bool(disable_selector)
        self.keyframe_topk = keyframe_topk
        self.keyframe_budget_ratio = float(keyframe_budget_ratio)

        if inbetween_ckpt is None:
            resolved_ckpt = os.path.abspath(_default_inbetween_ckpt_for_mode(selector_mode))
        else:
            resolved_ckpt = os.path.abspath(inbetween_ckpt)

        cfg = CompositeConfig(
            root=os.path.abspath(humanml_root),
            selector_mode=selector_mode,
        )
        if fallback_keyframe_strategy is not None:
            cfg.keyframe_strategy = fallback_keyframe_strategy
        cfg.keyframe_interval = int(keyframe_interval)
        cfg.keyframe_count = keyframe_count
        if keyframe_min is not None:
            cfg.keyframe_min = int(keyframe_min)
        if keyframe_max is not None:
            cfg.keyframe_max = int(keyframe_max)
        cfg.keyframe_include_ends = bool(include_ends)
        cfg.selector_target_ratio = float(keyframe_budget_ratio)
        if disable_selector and selector_mode == "reconstruction":
            cfg.use_learned_keyframe_selector = False
        self.cfg = cfg
        cache_key = (resolved_ckpt, str(device), os.path.abspath(humanml_root), int(ddim_steps))
        if cache_key not in _INBETWEEN_RUNTIME_CACHE:
            _INBETWEEN_RUNTIME_CACHE[cache_key] = load_inbetween_model(
                cfg,
                str(device),
                inbetween_ckpt_path=resolved_ckpt,
                ddim_steps=int(ddim_steps),
            )
        self.inbetween_model, self.diff_inbetween, self.clip_model, local_mean, local_std, self.fdim = _INBETWEEN_RUNTIME_CACHE[cache_key]
        self.local_mean = local_mean.to(device)
        self.local_std = local_std.to(device)
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        self.selector_model = None
        if selector_mode == "reconstruction":
            if not self.cfg.use_learned_keyframe_selector:
                self.selector_model = None
            else:
                self.selector_model = getattr(self.inbetween_model, "keyframe_selector", None)
        else:
            self.selector_model = build_keyframe_selector(
                mode=selector_mode,
                feature_dim=self.fdim,
                cond_dim=512,
                d_model=self.cfg.selector_d_model,
                n_layers=self.cfg.selector_layers,
                n_heads=self.cfg.selector_heads,
                dropout=self.cfg.selector_dropout,
                max_len=self.cfg.max_len + 10,
                threshold=self.cfg.selector_threshold,
                topk=keyframe_topk,
                budget_ratio=keyframe_budget_ratio,
            ).to(device)

    def _select_keyframes(
        self,
        ar_motion_norm: torch.Tensor,
        raw_motion: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        effective_length = int(ar_motion_norm.shape[0])
        selector_model = self.selector_model
        if selector_model is not None:
            selector_model.eval()
            selector_motion = ar_motion_norm if self.selector_mode == "reconstruction" else raw_motion
            selector_valid = torch.ones(1, effective_length, dtype=torch.bool, device=self.device)
            _, selector_mask_st = selector_model(
                selector_motion.unsqueeze(0),
                selector_valid,
                cond=cond,
            )
            keyframe_indices = torch.nonzero(selector_mask_st[0] > 0.5, as_tuple=False).squeeze(1)
            if keyframe_indices.numel() > 2:
                return keyframe_indices

        idx_list = select_keyframe_indices(
            length=effective_length,
            keyframe_interval=self.cfg.keyframe_interval,
            strategy=self.cfg.keyframe_strategy,
            keyframe_count=self.cfg.keyframe_count,
            keyframe_min=self.cfg.keyframe_min,
            keyframe_max=self.cfg.keyframe_max,
            include_ends=self.cfg.keyframe_include_ends,
        )
        return torch.tensor(idx_list, dtype=torch.long, device=self.device)

    def _encode_prompt_cached(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_cache.get(prompt)
        if cached is not None:
            return cached
        cond = encode_text(self.clip_model, [prompt])
        cond_uncond = encode_text(self.clip_model, [""])
        self._text_cache[prompt] = (cond, cond_uncond)
        return cond, cond_uncond

    def prepare_conditioning(self, prompt: str, ar_motion: GeneratedARMotion) -> CompositeConditioning:
        raw_tensor = torch.from_numpy(ar_motion.raw_motion).to(self.device).float()
        ar_motion_norm = (raw_tensor - self.local_mean) / (self.local_std + 1e-8)

        cond, cond_uncond = self._encode_prompt_cached(prompt)
        keyframe_indices = self._select_keyframes(ar_motion_norm, raw_tensor, cond)
        return CompositeConditioning(
            prompt=prompt,
            ar_motion_norm=ar_motion_norm,
            cond=cond,
            cond_uncond=cond_uncond,
            keyframe_indices=keyframe_indices,
        )

    def sample_batch_conditioned(self, conditionings: list[CompositeConditioning]) -> list[np.ndarray]:
        lengths = [int(c.ar_motion_norm.shape[0]) for c in conditionings]
        max_len = max(lengths)
        batch_size = len(conditionings)

        source_motion = torch.zeros(batch_size, max_len, self.fdim, device=self.device)
        valid_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        for i, (c, length) in enumerate(zip(conditionings, lengths)):
            source_motion[i, :length] = c.ar_motion_norm
            valid_mask[i, :length] = True

        max_kf = max(c.keyframe_indices.numel() for c in conditionings)
        keyframe_indices = torch.zeros(batch_size, max_kf, dtype=torch.long, device=self.device)
        keyframe_mask = torch.zeros(batch_size, max_kf, dtype=torch.bool, device=self.device)
        for i, c in enumerate(conditionings):
            n = c.keyframe_indices.numel()
            keyframe_indices[i, :n] = c.keyframe_indices
            keyframe_mask[i, :n] = True

        sampled_norm = self.diff_inbetween.sample_inbetween(
            model=self.inbetween_model,
            shape=(batch_size, max_len, self.fdim),
            cond=None,
            mask=valid_mask,
            keyframe_indices=keyframe_indices,
            keyframe_mask=keyframe_mask,
            guidance_scale=self.diff_guidance,
            source_motion=source_motion,
            text_prompt=[c.prompt for c in conditionings],
        )
        return [
            (sampled_norm[i, :lengths[i]] * (self.local_std + 1e-8) + self.local_mean)
            .detach().cpu().numpy().astype(np.float32)
            for i in range(batch_size)
        ]

    def sample_conditioned(self, conditioning: CompositeConditioning) -> np.ndarray:
        ar_motion_norm = conditioning.ar_motion_norm
        keyframe_indices = conditioning.keyframe_indices
        keyframes_norm = ar_motion_norm[keyframe_indices]

        time_steps = int(ar_motion_norm.shape[0])
        valid_mask = torch.ones(1, time_steps, dtype=torch.bool, device=self.device)
        keyframe_mask = torch.ones(1, keyframe_indices.numel(), dtype=torch.bool, device=self.device)
        sampled_norm = self.diff_inbetween.sample_inbetween(
            model=self.inbetween_model,
            shape=(1, time_steps, self.fdim),
            cond=conditioning.cond,
            mask=valid_mask,
            keyframes=keyframes_norm.unsqueeze(0),
            keyframe_indices=keyframe_indices.unsqueeze(0),
            keyframe_mask=keyframe_mask,
            guidance_scale=self.diff_guidance,
            cond_uncond=conditioning.cond_uncond,
            source_motion=ar_motion_norm.unsqueeze(0),
            text_prompt=conditioning.prompt,
        )[0]
        raw_motion = sampled_norm * (self.local_std + 1e-8) + self.local_mean
        return raw_motion.detach().cpu().numpy().astype(np.float32)

    def generate_from_ar(self, prompt: str, ar_motion: GeneratedARMotion) -> np.ndarray:
        return self.sample_conditioned(self.prepare_conditioning(prompt, ar_motion))


class CondMDITextOnlyStrategy:
    def __init__(
        self,
        name: str,
        humanml_root: str,
        device: torch.device,
        checkpoint_path: str,
        guidance_scale: float,
        ddim_steps: int = 50,
    ):
        self.name = name
        self.device = device
        self.guidance_scale = float(guidance_scale)

        mean_path = os.path.join(humanml_root, "Mean.npy")
        std_path = os.path.join(humanml_root, "Std.npy")
        local_mean = torch.from_numpy(np.load(mean_path)).float().view(-1)
        local_std = torch.from_numpy(np.load(std_path)).float().view(-1)
        self.local_mean = local_mean.to(device)
        self.local_std = local_std.to(device)
        self.fdim = int(local_mean.shape[0])

        resolved_ckpt = os.path.abspath(checkpoint_path)
        args_path = os.path.join(os.path.dirname(resolved_ckpt), "args.json")
        if not os.path.exists(resolved_ckpt) or not os.path.exists(args_path):
            raise FileNotFoundError(
                "CondMDI unconditional checkpoint is not set up. "
                "Download the CondMDI unconditional model and place both "
                f"{os.path.basename(resolved_ckpt)} and args.json in {os.path.dirname(resolved_ckpt)}, "
                "or update EvalConfig.condmdi_unconditional_ckpt."
            )
        cache_key = (resolved_ckpt, str(device), os.path.abspath(humanml_root), int(ddim_steps))
        if cache_key not in _CONDMDI_TEXT_RUNTIME_CACHE:
            model, adapter = load_external_condmdi_runtime(
                checkpoint_path=resolved_ckpt,
                local_mean=local_mean,
                local_std=local_std,
                device=str(device),
                ddim_steps=int(ddim_steps),
            )
            _CONDMDI_TEXT_RUNTIME_CACHE[cache_key] = (model, adapter, local_mean, local_std, self.fdim)
            print(f"Loaded CondMDI text-only runtime from {resolved_ckpt}")
        self.model, self.adapter, _, _, _ = _CONDMDI_TEXT_RUNTIME_CACHE[cache_key]

    def sample_batch(self, prompts: list[str], lengths: torch.Tensor) -> list[np.ndarray]:
        sampled_norm = self.adapter.sample_text_local(
            text_prompts=prompts,
            lengths=lengths.to(self.device),
            guidance_scale=self.guidance_scale,
        )
        raw_motion = sampled_norm * (self.local_std + 1e-8) + self.local_mean
        return [
            raw_motion[i, : int(lengths[i].item())].detach().cpu().numpy().astype(np.float32)
            for i in range(raw_motion.shape[0])
        ]


def build_motion_batch(
    raw_motions_in: Iterable[np.ndarray],
    eval_mean: np.ndarray,
    eval_std: np.ndarray,
    need_jerk: bool = False,
    need_foot_skating: bool = False,
) -> MotionBatch:
    raw_motions: list[np.ndarray] = []
    jerk_scores: list[float] = []
    foot_scores: list[float] = []
    padded = []
    padded_lengths = []
    for raw_motion in raw_motions_in:
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


class MetricAccumulator:
    def __init__(self, model_name: str, metrics: set[str], top_k: list[int]):
        self.model_name = model_name
        self.metrics = metrics
        self.top_k = top_k
        self.max_k = max(top_k)
        self.processed = 0
        self.r_precision_sums = np.zeros(self.max_k, dtype=np.float64)
        self.matching_score_total = 0.0
        self.gt_motion_embeddings: list[np.ndarray] = []
        self.pred_motion_embeddings: list[np.ndarray] = []
        self.jerk_scores: list[float] = []
        self.foot_scores: list[float] = []

    def update_first_pass(
        self,
        text_embeddings: torch.Tensor,
        gt_embeddings: torch.Tensor,
        pred_embeddings: torch.Tensor,
        motion_batch: MotionBatch,
    ) -> None:
        batch_size = int(pred_embeddings.shape[0])
        if "r_precision" in self.metrics or "matching_score" in self.metrics:
            temp_r, temp_match = calculate_r_precision(
                text_embeddings.detach().cpu().numpy(),
                pred_embeddings.detach().cpu().numpy(),
                max_k=self.max_k,
            )
            self.r_precision_sums += temp_r
            self.matching_score_total += temp_match

        self.gt_motion_embeddings.append(gt_embeddings.detach().cpu().numpy())
        self.pred_motion_embeddings.append(pred_embeddings.detach().cpu().numpy())
        if "jerk" in self.metrics:
            self.jerk_scores.extend(motion_batch.jerk_scores)
        if "foot_skating" in self.metrics:
            self.foot_scores.extend(motion_batch.foot_skating_scores)
        self.processed += batch_size

    def finalize(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "model": self.model_name,
            "num_samples": int(self.processed),
        }

        gt_motion_np = np.concatenate(self.gt_motion_embeddings, axis=0)
        pred_motion_np = np.concatenate(self.pred_motion_embeddings, axis=0)

        if "fid" in self.metrics:
            gt_mu, gt_cov = calculate_activation_statistics(gt_motion_np)
            pred_mu, pred_cov = calculate_activation_statistics(pred_motion_np)
            results["fid"] = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)

        if "diversity" in self.metrics:
            diversity_times = 300 if pred_motion_np.shape[0] > 300 else max(1, pred_motion_np.shape[0] - 1)
            results["diversity"] = calculate_diversity(pred_motion_np, diversity_times)

        if "matching_score" in self.metrics:
            results["matching_score"] = float(self.matching_score_total / max(self.processed, 1))

        if "r_precision" in self.metrics:
            r_precision = {}
            for k in self.top_k:
                r_precision[f"R@{k}"] = float(self.r_precision_sums[k - 1] / max(self.processed, 1))
            results["r_precision"] = r_precision

        if "jerk" in self.metrics:
            results["jerk"] = float(np.mean(self.jerk_scores)) if self.jerk_scores else 0.0

        if "foot_skating" in self.metrics:
            results["foot_skating"] = float(np.mean(self.foot_scores)) if self.foot_scores else 0.0

        return results


def evaluate_models(
    models: list[str],
    ar_source: T2MGPTMotionSource,
    composite_strategies: dict[str, CompositeStrategy],
    condmdi_text_strategies: dict[str, CondMDITextOnlyStrategy],
    loader: DataLoader,
    eval_wrapper,
    top_k: list[int],
    metrics: set[str],
    num_samples: int,
    categorical_sample: bool,
) -> dict[str, dict[str, Any]]:
    device = next(eval_wrapper.motion_encoder.parameters()).device
    eval_mean = loader.dataset.mean.astype(np.float32)
    eval_std = loader.dataset.std.astype(np.float32)
    accumulators = {
        model_name: MetricAccumulator(model_name, metrics, top_k)
        for model_name in models
    }
    processed = 0

    with tqdm(total=num_samples * len(models), unit="pass", desc="Evaluating") as pbar:
        for batch in loader:
            remaining = num_samples - processed
            if remaining <= 0:
                break
            word_embeddings, pos_one_hots, captions, sent_len, motion, m_length, _, _ = batch
            batch_size = len(captions)
            if batch_size > remaining:
                batch = truncate_batch(batch, remaining)
                word_embeddings, pos_one_hots, captions, sent_len, motion, m_length, _, _ = batch
                batch_size = remaining

            captions = list(captions)
            motion = motion.to(device).float()
            m_length = m_length.to(device).long()

            pbar.set_postfix(model="gt_embed", refresh=True)
            text_embeddings, gt_embeddings = eval_wrapper.get_co_embeddings(
                word_embeddings,
                pos_one_hots,
                sent_len,
                motion,
                m_length,
            )

            pbar.set_postfix(model="t2mgpt_gen", refresh=True)
            base_ar_motions = [
                ar_source.generate(caption, stochastic=bool(categorical_sample))
                for caption in captions
            ]

            model_bar = tqdm(models, desc="  models", leave=False, unit="model")
            for model_name in model_bar:
                model_bar.set_description(f"  {model_name}")
                pbar.set_postfix(model=model_name, refresh=True)
                if model_name == "t2mgpt":
                    raw_motions = [motion.raw_motion for motion in base_ar_motions]
                elif model_name in condmdi_text_strategies:
                    raw_motions = condmdi_text_strategies[model_name].sample_batch(captions, m_length)
                else:
                    strategy = composite_strategies[model_name]
                    conditionings = [
                        strategy.prepare_conditioning(prompt=caption, ar_motion=ar_motion)
                        for caption, ar_motion in zip(captions, base_ar_motions)
                    ]
                    raw_motions = strategy.sample_batch_conditioned(conditionings)

                motion_batch = build_motion_batch(
                    raw_motions,
                    eval_mean=eval_mean,
                    eval_std=eval_std,
                    need_jerk=("jerk" in metrics),
                    need_foot_skating=("foot_skating" in metrics),
                )
                pred_padded = motion_batch.padded.to(device)
                pred_lengths = motion_batch.lengths.to(device)
                _, pred_embeddings = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    pred_padded,
                    pred_lengths,
                )
                accumulators[model_name].update_first_pass(
                    text_embeddings=text_embeddings,
                    gt_embeddings=gt_embeddings,
                    pred_embeddings=pred_embeddings,
                    motion_batch=motion_batch,
                )
                pbar.update(batch_size)

            processed += batch_size
            pbar.set_postfix(model="—", refresh=True)

    return {
        model_name: accumulator.finalize()
        for model_name, accumulator in accumulators.items()
    }


def print_results(all_results: dict[str, Any]) -> None:
    print("\nEvaluation summary")
    print(json.dumps(all_results, indent=2, sort_keys=True))


def _flatten_metric_value(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _flatten_metric_value(f"{prefix}.{key}", nested_value, out)
    else:
        out[prefix] = value


def write_results_csv(all_results: dict[str, Any], csv_path: str) -> None:
    new_rows: list[dict[str, Any]] = []
    metrics = ",".join(all_results.get("metrics", []))
    for model_name, result in all_results.get("models", {}).items():
        row: dict[str, Any] = {
            "model": model_name,
            "num_samples": result.get("num_samples", all_results.get("num_samples")),
            "metrics": metrics,
            "timestamp": all_results.get("timestamp", ""),
            "device": all_results.get("device", ""),
            "keyframe_topk": all_results.get("keyframe_topk", ""),
            "keyframe_budget_ratio": all_results.get("keyframe_budget_ratio", ""),
        }
        for key, value in result.items():
            if key in {"model", "num_samples"}:
                continue
            _flatten_metric_value(key, value, row)
        new_rows.append(row)

    if not new_rows:
        return

    existing_rows: list[dict[str, Any]] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    rows_by_model = {
        row.get("model", ""): row
        for row in existing_rows
        if row.get("model", "")
    }
    model_order = [
        row.get("model", "")
        for row in existing_rows
        if row.get("model", "")
    ]

    for row in new_rows:
        model_name = str(row.get("model", ""))
        if model_name not in rows_by_model:
            model_order.append(model_name)
        rows_by_model[model_name] = row

    rows = [rows_by_model[model_name] for model_name in model_order if model_name in rows_by_model]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "model",
        "num_samples",
        "fid",
        "diversity",
        "matching_score",
        "r_precision.R@1",
        "r_precision.R@2",
        "r_precision.R@3",
        "r_precision.R@5",
        "jerk",
        "foot_skating",
    ]
    ordered = [key for key in preferred if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _default_inbetween_ckpt_for_mode(selector_mode: str) -> str:
    mode = str(selector_mode).strip().lower().replace("-", "_")
    if mode == "reconstruction":
        preferred_selector = [
            os.path.join("checkpoints", "composite_selector_reconstruction_best.pt"),
        ]
        for path in preferred_selector:
            if os.path.exists(path):
                return path
        selector_steps = glob.glob(os.path.join("checkpoints", "composite_selector_reconstruction_step*.pt"))
        if selector_steps:
            def _extract_selector_step(path: str) -> int:
                stem = os.path.basename(path).removesuffix(".pt")
                step_text = stem.removeprefix("composite_selector_reconstruction_step")
                return int(step_text) if step_text.isdigit() else -1

            return max(selector_steps, key=_extract_selector_step)

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

    return _default_inbetween_ckpt()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(cfg.seed)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t2mgpt_root = os.path.abspath(cfg.t2mgpt_root)
    results_path = ensure_results_path(cfg)
    csv_path = ensure_csv_path(cfg, results_path)

    models = resolve_models(cfg.models)
    metrics = resolve_metrics(cfg.metrics)
    top_k = parse_top_k(cfg.r_precision_top_k)
    if cfg.num_samples <= 0:
        raise ValueError("EvalConfig.num_samples must be positive")
    if cfg.keyframe_topk is not None and cfg.keyframe_topk <= 0:
        raise ValueError("EvalConfig.keyframe_topk must be positive when set")
    if cfg.keyframe_budget_ratio <= 0:
        raise ValueError("EvalConfig.keyframe_budget_ratio must be positive")

    loader, eval_wrapper = build_eval_loader_and_wrapper(
        t2mgpt_root=t2mgpt_root,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    ar_source = T2MGPTMotionSource(
        t2mgpt_root=t2mgpt_root,
        device=device,
        arlm_vq_ckpt=cfg.arlm_vq_ckpt,
        arlm_gpt_ckpt=cfg.arlm_gpt_ckpt,
        clamp_sigma=cfg.ar_clamp_sigma,
    )

    composite_strategies: dict[str, CompositeStrategy] = {}
    condmdi_text_strategies: dict[str, CondMDITextOnlyStrategy] = {}
    if CONDMDI_UNCONDITIONAL_MODEL in models:
        condmdi_text_strategies[CONDMDI_UNCONDITIONAL_MODEL] = CondMDITextOnlyStrategy(
            name=CONDMDI_UNCONDITIONAL_MODEL,
            humanml_root=cfg.humanml_root,
            device=device,
            checkpoint_path=cfg.condmdi_unconditional_ckpt,
            guidance_scale=cfg.diff_guidance,
            ddim_steps=cfg.ddim_steps,
        )

    for model_name in models:
        if not model_name.startswith("composite_"):
            continue
        selector_mode = model_name.removeprefix("composite_")
        ckpt_override = cfg.composite_reconstruction_ckpt if selector_mode == "reconstruction" else None
        inbetween_ckpt = cfg.composite_inbetween_ckpt or ckpt_override
        disable_selector = bool(cfg.disable_reconstruction_selector and selector_mode == "reconstruction")

        composite_strategies[model_name] = CompositeStrategy(
            name=model_name,
            humanml_root=cfg.humanml_root,
            device=device,
            inbetween_ckpt=inbetween_ckpt,
            selector_mode=selector_mode,
            disable_selector=disable_selector,
            diff_guidance=cfg.diff_guidance,
            fallback_keyframe_strategy=cfg.fallback_keyframe_strategy,
            keyframe_interval=cfg.keyframe_interval,
            keyframe_count=cfg.keyframe_count,
            keyframe_min=cfg.keyframe_min,
            keyframe_max=cfg.keyframe_max,
            include_ends=cfg.keyframe_include_ends,
            keyframe_topk=cfg.keyframe_topk,
            keyframe_budget_ratio=cfg.keyframe_budget_ratio,
            ddim_steps=cfg.ddim_steps,
        )

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "metrics": sorted(metrics),
        "models": {},
        "num_samples": int(cfg.num_samples),
        "keyframe_topk": cfg.keyframe_topk,
        "keyframe_budget_ratio": float(cfg.keyframe_budget_ratio),
    }

    print(f"Running evaluation for {', '.join(models)} on {device}")
    with torch.inference_mode():
        all_results["models"] = evaluate_models(
            models=models,
            ar_source=ar_source,
            composite_strategies=composite_strategies,
            condmdi_text_strategies=condmdi_text_strategies,
            loader=loader,
            eval_wrapper=eval_wrapper,
            top_k=top_k,
            metrics=metrics,
            num_samples=cfg.num_samples,
            categorical_sample=cfg.categorical_sample,
        )

    if cfg.save_json:
        results_dir = os.path.dirname(results_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved results to {results_path}")

    write_results_csv(all_results, csv_path)
    print(f"Saved CSV results to {csv_path}")
    print_results(all_results)


if __name__ == "__main__":
    main()
