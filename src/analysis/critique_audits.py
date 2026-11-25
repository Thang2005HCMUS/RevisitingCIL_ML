import copy
import csv
import json
import logging
import math
import os
import statistics
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from analysis.sensitivity_sweep import (
    _l2_normalize_rows,
    build_weighted_features,
    evaluate_cil_trajectory,
    load_feature_cache,
)
from framework.learner import _is_wandb_enabled, _resolve_wandb_tags
from models.classifiers import CosinePrototypeClassifier
from models.merged_encoder import DualBranchEncoder, _get_module_out_dim, _validate_alpha
from utils.metrics import count_parameters

DEFAULT_CRITIQUE_WANDB_TAGS: Tuple[str, ...] = ("empirics", "critique-audit")

_REQUIRED_NORM_CACHE_KEYS: Tuple[str, ...] = (
    "train_adapted",
    "train_frozen",
    "train_targets",
    "test_adapted",
    "test_frozen",
    "test_targets",
)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that value is a strictly positive integer (excluding booleans)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        int_val = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int_val <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int_val


def _validate_positive_float(value: Any, name: str) -> float:
    """Validate that value is a finite strictly positive float (excluding booleans)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float, got {value!r}.")
    try:
        float_val = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}.") from exc
    if math.isnan(float_val) or math.isinf(float_val) or float_val <= 0.0:
        raise ValueError(f"{name} must be a positive finite float, got {value!r}.")
    return float_val


def _to_numpy_2d(arr: Any, name: str) -> np.ndarray:
    """Convert tensor/array-like to a contiguous 2D float64 NumPy array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D array for '{name}', got shape {out.shape}.")
    return out


def _to_numpy_1d_int(arr: Any, name: str) -> np.ndarray:
    """Convert tensor/array-like to a 1D int64 NumPy array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    out = np.asarray(arr, dtype=np.int64)
    if out.ndim != 1 or out.shape[0] == 0:
        raise ValueError(f"Expected non-empty 1D array for '{name}', got shape {out.shape}.")
    return out


def _summarize_distribution(values: np.ndarray) -> Dict[str, float]:
    """Compute mean, std, median, p05, and p95 summary metrics for a 1D array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _compute_branch_norm_metrics(
    adapted: np.ndarray,
    frozen: np.ndarray,
    eps: float,
) -> Dict[str, Any]:
    """Compute per-row L2 norms, norm ratio r, and implicit mixing weight alpha_eff."""
    adapted_norms = np.linalg.norm(adapted, axis=-1)
    frozen_norms = np.linalg.norm(frozen, axis=-1)

    zero_adapted_count = int(np.sum(adapted_norms <= eps))
    zero_frozen_count = int(np.sum(frozen_norms <= eps))

    norm_ratio = adapted_norms / np.maximum(frozen_norms, eps)
    adapted_sq = adapted_norms ** 2
    frozen_sq = frozen_norms ** 2
    denom_sq = np.maximum(adapted_sq + frozen_sq, eps)
    alpha_eff = np.clip(adapted_sq / denom_sq, 0.0, 1.0)

    return {
        "n_samples": int(adapted.shape[0]),
        "zero_adapted_count": zero_adapted_count,
        "zero_frozen_count": zero_frozen_count,
        "adapted_norm": _summarize_distribution(adapted_norms),
        "frozen_norm": _summarize_distribution(frozen_norms),
        "norm_ratio": _summarize_distribution(norm_ratio),
        "alpha_eff": _summarize_distribution(alpha_eff),
    }


def _validate_cached_features_dict(
    cached_features: Any,
    require_increments: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate required fields and shapes of an Exp 3 dual-branch feature cache."""
    if not isinstance(cached_features, dict):
        raise ValueError(f"cached_features must be a dict, got {type(cached_features).__name__}.")

    required = list(_REQUIRED_NORM_CACHE_KEYS)
    if require_increments:
        required.append("increments")
    missing = [key for key in required if key not in cached_features]
    if missing:
        raise ValueError(f"Feature cache is missing required keys: {missing}")

    train_adapted = _to_numpy_2d(cached_features["train_adapted"], "train_adapted")
    train_frozen = _to_numpy_2d(cached_features["train_frozen"], "train_frozen")
    train_targets = _to_numpy_1d_int(cached_features["train_targets"], "train_targets")
    test_adapted = _to_numpy_2d(cached_features["test_adapted"], "test_adapted")
    test_frozen = _to_numpy_2d(cached_features["test_frozen"], "test_frozen")
    test_targets = _to_numpy_1d_int(cached_features["test_targets"], "test_targets")

    if train_adapted.shape != train_frozen.shape:
        raise ValueError(
            f"train_adapted shape {train_adapted.shape} does not match train_frozen shape {train_frozen.shape}."
        )
    if test_adapted.shape != test_frozen.shape:
        raise ValueError(
            f"test_adapted shape {test_adapted.shape} does not match test_frozen shape {test_frozen.shape}."
        )
    if train_adapted.shape[1] != test_adapted.shape[1]:
        raise ValueError(
            f"Train feature dim {train_adapted.shape[1]} does not match test feature dim {test_adapted.shape[1]}."
        )
    if train_targets.shape[0] != train_adapted.shape[0]:
        raise ValueError(
            f"train_targets length {train_targets.shape[0]} does not match train_adapted rows {train_adapted.shape[0]}."
        )
    if test_targets.shape[0] != test_adapted.shape[0]:
        raise ValueError(
            f"test_targets length {test_targets.shape[0]} does not match test_adapted rows {test_adapted.shape[0]}."
        )
    return train_adapted, train_frozen, train_targets, test_adapted, test_frozen, test_targets


def compute_norm_imbalance_stats(
    cached_features: Dict[str, Any],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute per-sample L2 norm distributions, norm ratio r = ||phi*(x)||_2 / ||phi(x)||_2,
    effective implicit mixing weight alpha_eff = ||phi*(x)||_2^2 / (||phi*(x)||_2^2 + ||phi(x)||_2^2),
    and class-prototype norm ratios across train and test splits without retraining.
    """
    eps_val = _validate_positive_float(eps, "eps")
    (
        train_adapted,
        train_frozen,
        train_targets,
        test_adapted,
        test_frozen,
        _,
    ) = _validate_cached_features_dict(cached_features, require_increments=False)

    train_stats = _compute_branch_norm_metrics(train_adapted, train_frozen, eps_val)
    test_stats = _compute_branch_norm_metrics(test_adapted, test_frozen, eps_val)

    task_class_ids = cached_features.get("task_class_ids")
    if task_class_ids:
        ordered_classes = [int(c) for cids in task_class_ids for c in cids]
    else:
        ordered_classes = [int(c) for c in np.unique(train_targets)]

    proto_adapted_list: List[np.ndarray] = []
    proto_frozen_list: List[np.ndarray] = []
    for cls_id in ordered_classes:
        cls_mask = train_targets == cls_id
        if np.any(cls_mask):
            proto_adapted_list.append(np.mean(train_adapted[cls_mask], axis=0))
            proto_frozen_list.append(np.mean(train_frozen[cls_mask], axis=0))

    if proto_adapted_list:
        proto_adapted = np.stack(proto_adapted_list, axis=0)
        proto_frozen = np.stack(proto_frozen_list, axis=0)
        proto_metrics = _compute_branch_norm_metrics(proto_adapted, proto_frozen, eps_val)
    else:
        proto_metrics = _compute_branch_norm_metrics(
            np.empty((0, train_adapted.shape[1]), dtype=np.float64),
            np.empty((0, train_frozen.shape[1]), dtype=np.float64),
            eps_val,
        )
    proto_stats = {
        "num_classes": int(len(proto_adapted_list)),
        "zero_adapted_count": proto_metrics["zero_adapted_count"],
        "zero_frozen_count": proto_metrics["zero_frozen_count"],
        "adapted_norm": proto_metrics["adapted_norm"],
        "frozen_norm": proto_metrics["frozen_norm"],
        "norm_ratio": proto_metrics["norm_ratio"],
        "alpha_eff": proto_metrics["alpha_eff"],
    }

    metadata = copy.deepcopy(cached_features.get("metadata") or {})
    return {
        "eps": eps_val,
        "feature_dim": int(train_adapted.shape[1]),
        "metadata": metadata,
        "train": train_stats,
        "test": test_stats,
        "prototypes": proto_stats,
        "train_mean_norm_ratio": train_stats["norm_ratio"]["mean"],
        "train_mean_alpha_eff": train_stats["alpha_eff"]["mean"],
        "test_mean_norm_ratio": test_stats["norm_ratio"]["mean"],
        "test_mean_alpha_eff": test_stats["alpha_eff"]["mean"],
        "prototype_mean_norm_ratio": proto_stats["norm_ratio"]["mean"],
        "prototype_mean_alpha_eff": proto_stats["alpha_eff"]["mean"],
    }


def run_norm_merge_audit(
    cached_features: Union[str, os.PathLike, Dict[str, Any]],
    best_alpha: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Evaluate side-by-side continual learning trajectories on identical cached features comparing
    unnormalized concatenation ([phi*(x), phi(x)]) against branch-normalized alpha-weighted merges.
    """
    best_alpha_val = _validate_alpha(best_alpha)
    eps_val = _validate_positive_float(eps, "eps")

    cache_path: Optional[str] = None
    if isinstance(cached_features, (str, os.PathLike)):
        cache_path = os.fspath(cached_features)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Feature cache file not found at '{cache_path}'. "
                "Provide a valid Exp 3 feature cache (.pt) path."
            )
        data = load_feature_cache(cache_path)
    else:
        data = cached_features

    (
        train_adapted_f64,
        train_frozen_f64,
        train_targets,
        test_adapted_f64,
        test_frozen_f64,
        test_targets,
    ) = _validate_cached_features_dict(data, require_increments=True)

    increments = [int(x) for x in data["increments"]]
    if not increments or any(x <= 0 for x in increments):
        raise ValueError(f"increments must contain positive integers, got {increments}.")
    task_class_ids = data.get("task_class_ids")

    norm_stats = compute_norm_imbalance_stats(data, eps=eps_val)

    train_adapted_f32 = train_adapted_f64.astype(np.float32, copy=False)
    train_frozen_f32 = train_frozen_f64.astype(np.float32, copy=False)
    test_adapted_f32 = test_adapted_f64.astype(np.float32, copy=False)
    test_frozen_f32 = test_frozen_f64.astype(np.float32, copy=False)

    tr_ad_norm = _l2_normalize_rows(train_adapted_f32, eps=eps_val)
    tr_fr_norm = _l2_normalize_rows(train_frozen_f32, eps=eps_val)
    te_ad_norm = _l2_normalize_rows(test_adapted_f32, eps=eps_val)
    te_fr_norm = _l2_normalize_rows(test_frozen_f32, eps=eps_val)

    variant_specs = [
        (
            "unnormalized_concat",
            "Unnormalized Concat [phi*, phi]",
            False,
            None,
            float(norm_stats["test_mean_alpha_eff"]),
            np.concatenate([train_adapted_f32, train_frozen_f32], axis=-1),
            np.concatenate([test_adapted_f32, test_frozen_f32], axis=-1),
        ),
        (
            "normalized_alpha_0.5",
            "Normalized Merge (alpha=0.5)",
            True,
            0.5,
            0.5,
            build_weighted_features(tr_ad_norm, tr_fr_norm, 0.5),
            build_weighted_features(te_ad_norm, te_fr_norm, 0.5),
        ),
        (
            "normalized_best_alpha",
            f"Normalized Merge (alpha={best_alpha_val:g})",
            True,
            best_alpha_val,
            best_alpha_val,
            build_weighted_features(tr_ad_norm, tr_fr_norm, best_alpha_val),
            build_weighted_features(te_ad_norm, te_fr_norm, best_alpha_val),
        ),
        (
            "ptm_only",
            "Frozen PTM Only (alpha=0.0)",
            True,
            0.0,
            0.0,
            build_weighted_features(tr_ad_norm, tr_fr_norm, 0.0),
            build_weighted_features(te_ad_norm, te_fr_norm, 0.0),
        ),
        (
            "adapted_only",
            "Adapted Only (alpha=1.0)",
            True,
            1.0,
            1.0,
            build_weighted_features(tr_ad_norm, tr_fr_norm, 1.0),
            build_weighted_features(te_ad_norm, te_fr_norm, 1.0),
        ),
    ]

    variants: List[Dict[str, Any]] = []
    for variant_key, label, is_norm, alpha_cfg, alpha_eff, tr_feat, te_feat in variant_specs:
        traj = evaluate_cil_trajectory(
            tr_feat,
            train_targets,
            te_feat,
            test_targets,
            increments=increments,
            task_class_ids=task_class_ids,
        )
        variants.append(
            {
                "variant": variant_key,
                "label": label,
                "normalized": bool(is_norm),
                "alpha_config": alpha_cfg,
                "effective_alpha": float(alpha_eff),
                "avg_acc": float(traj["avg_acc"]),
                "last_acc": float(traj["last_acc"]),
                "old_acc": float(traj["old_acc"]),
                "new_acc": float(traj["new_acc"]),
                "last_old_acc": float(traj["last_old_acc"]),
                "last_new_acc": float(traj["last_new_acc"]),
                "top1_curve": traj["top1_curve"],
                "old_curve": traj["old_curve"],
                "new_curve": traj["new_curve"],
            }
        )

    baseline = variants[0]
    for row in variants:
        row["delta_avg_acc_vs_unnorm"] = round(row["avg_acc"] - baseline["avg_acc"], 2)
        row["delta_last_acc_vs_unnorm"] = round(row["last_acc"] - baseline["last_acc"], 2)
        row["delta_old_acc_vs_unnorm"] = round(row["old_acc"] - baseline["old_acc"], 2)
        row["delta_new_acc_vs_unnorm"] = round(row["new_acc"] - baseline["new_acc"], 2)

    variants_by_name = {row["variant"]: row for row in variants}
    metadata = copy.deepcopy(data.get("metadata") or {})
    dataset = str(metadata.get("dataset") or "unknown")
    best_norm_variant = max(
        [v for v in variants if v["normalized"]],
        key=lambda r: (r["avg_acc"], r["last_acc"]),
    )

    return {
        "dataset": dataset,
        "cache_path": cache_path,
        "metadata": metadata,
        "increments": increments,
        "best_alpha": best_alpha_val,
        "norm_stats": norm_stats,
        "variants": variants,
        "variants_by_name": variants_by_name,
        "summary": {
            "dataset": dataset,
            "train_mean_norm_ratio": norm_stats["train_mean_norm_ratio"],
            "train_mean_alpha_eff": norm_stats["train_mean_alpha_eff"],
            "test_mean_norm_ratio": norm_stats["test_mean_norm_ratio"],
            "test_mean_alpha_eff": norm_stats["test_mean_alpha_eff"],
            "prototype_mean_norm_ratio": norm_stats["prototype_mean_norm_ratio"],
            "prototype_mean_alpha_eff": norm_stats["prototype_mean_alpha_eff"],
            "unnormalized_avg_acc": variants_by_name["unnormalized_concat"]["avg_acc"],
            "normalized_alpha_0_5_avg_acc": variants_by_name["normalized_alpha_0.5"]["avg_acc"],
            "normalized_best_alpha_avg_acc": variants_by_name["normalized_best_alpha"]["avg_acc"],
            "delta_alpha_0_5_vs_unnorm": variants_by_name["normalized_alpha_0.5"]["delta_avg_acc_vs_unnorm"],
            "delta_best_alpha_vs_unnorm": variants_by_name["normalized_best_alpha"]["delta_avg_acc_vs_unnorm"],
            "best_normalized_variant": best_norm_variant["variant"],
        },
    }


class _InferencePipeline(nn.Module):
    """Lightweight inference wrapper combining DualBranchEncoder and CosinePrototypeClassifier."""

    def __init__(self, encoder: DualBranchEncoder, classifier: CosinePrototypeClassifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


def _measure_module_footprint_mb(module: nn.Module) -> float:
    """Compute parameter + buffer memory footprint of a PyTorch module in MiB."""
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
    return float(total_bytes) / (1024.0 ** 2)


def benchmark_inference_variants(
    frozen_ptm: nn.Module,
    adapted_ptm: nn.Module,
    batch_size: int = 64,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_classes: int = 100,
    pca_dim: int = 128,
    warmup_iters: int = 10,
    timed_iters: int = 50,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_mode: str = "identical",
) -> Dict[str, Any]:
    """
    Benchmark wall-clock forward-pass latency, throughput, parameter count, and peak memory
    across SimpleCIL (1-branch), APER Full (2-branch), and APER + PCA (2-branch + linear k).
    """
    if frozen_ptm is None or adapted_ptm is None:
        raise ValueError("Both frozen_ptm and adapted_ptm must be provided.")

    batch_size_val = _validate_positive_int(batch_size, "batch_size")
    num_classes_val = _validate_positive_int(num_classes, "num_classes")
    pca_dim_val = _validate_positive_int(pca_dim, "pca_dim")
    warmup_val = _validate_positive_int(warmup_iters, "warmup_iters")
    timed_val = _validate_positive_int(timed_iters, "timed_iters")

    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3-tuple (C, H, W), got {input_shape!r}.")
    shape_tuple = tuple(_validate_positive_int(dim, f"input_shape[{idx}]") for idx, dim in enumerate(input_shape))

    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    frozen_ptm.to(resolved_device).eval()
    adapted_ptm.to(resolved_device).eval()
    for param in frozen_ptm.parameters():
        param.requires_grad = False
    for param in adapted_ptm.parameters():
        param.requires_grad = False

    frozen_dim = _get_module_out_dim(frozen_ptm)
    adapted_dim = _get_module_out_dim(adapted_ptm)
    merged_dim = frozen_dim + adapted_dim

    def _build_classifier(feat_dim: int, seed: int) -> CosinePrototypeClassifier:
        clf = CosinePrototypeClassifier(feat_dim)
        gen = torch.Generator().manual_seed(seed)
        protos = torch.randn(num_classes_val, feat_dim, generator=gen)
        clf.register_prototypes(protos)
        clf.to(resolved_device).eval()
        for param in clf.parameters():
            param.requires_grad = False
        return clf

    simple_encoder = DualBranchEncoder(frozen_ptm, None, merge_mode="ptm_only", weighted_norm=True)
    aper_encoder = DualBranchEncoder(frozen_ptm, adapted_ptm, merge_mode="both", weighted_norm=True, alpha=0.5)

    pca_projector = nn.Linear(merged_dim, pca_dim_val, bias=False)
    gen_proj = torch.Generator().manual_seed(42)
    with torch.no_grad():
        pca_projector.weight.copy_(torch.randn(pca_dim_val, merged_dim, generator=gen_proj) / math.sqrt(merged_dim))
    aper_pca_encoder = DualBranchEncoder(
        frozen_ptm,
        adapted_ptm,
        merge_mode="both",
        weighted_norm=True,
        alpha=0.5,
        projector=pca_projector,
        projected_dim=pca_dim_val,
    )

    variant_configs = [
        ("simplecil", f"SimpleCIL (1-branch, {frozen_dim}d)", 1, frozen_dim, simple_encoder, _build_classifier(frozen_dim, 101)),
        ("aper_full", f"APER Full (2-branch, {merged_dim}d)", 2, merged_dim, aper_encoder, _build_classifier(merged_dim, 102)),
        ("aper_pca", f"APER + PCA (2-branch + {pca_dim_val}d)", 2, pca_dim_val, aper_pca_encoder, _build_classifier(pca_dim_val, 103)),
    ]

    x_batch = torch.randn(batch_size_val, *shape_tuple, device=resolved_device)
    is_cuda = resolved_device.type == "cuda"
    memory_backend = "cuda_max_memory_allocated" if is_cuda else "tracemalloc_mb"

    rows: List[Dict[str, Any]] = []
    for variant_key, label, branches, feat_dim, encoder, classifier in variant_configs:
        pipeline = _InferencePipeline(encoder, classifier).to(resolved_device).eval()
        for param in pipeline.parameters():
            param.requires_grad = False

        with torch.inference_mode():
            for _ in range(warmup_val):
                _ = pipeline(x_batch)
            if is_cuda:
                torch.cuda.synchronize(resolved_device)

        was_tracing = False
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(resolved_device)
        else:
            was_tracing = tracemalloc.is_tracing()
            if was_tracing:
                tracemalloc.reset_peak()
            else:
                tracemalloc.start()

        iter_ms: List[float] = []
        with torch.inference_mode():
            for _ in range(timed_val):
                if is_cuda:
                    torch.cuda.synchronize(resolved_device)
                t0 = time.perf_counter()
                _ = pipeline(x_batch)
                if is_cuda:
                    torch.cuda.synchronize(resolved_device)
                t1 = time.perf_counter()
                iter_ms.append((t1 - t0) * 1000.0)

        if is_cuda:
            peak_memory_mb = float(torch.cuda.max_memory_allocated(resolved_device)) / (1024.0 ** 2)
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_memory_mb = float(peak_bytes) / (1024.0 ** 2)
            if not was_tracing:
                tracemalloc.stop()

        ms_batch_mean = float(statistics.mean(iter_ms))
        ms_batch_std = float(statistics.stdev(iter_ms)) if len(iter_ms) > 1 else 0.0
        ms_img_mean = ms_batch_mean / float(batch_size_val)
        ms_img_std = ms_batch_std / float(batch_size_val)
        images_per_sec = (float(batch_size_val) * 1000.0) / max(ms_batch_mean, 1e-12)

        encoder_params = int(count_parameters(encoder, trainable=False))
        classifier_params = int(count_parameters(classifier, trainable=False))
        total_params = int(count_parameters(pipeline, trainable=False))
        trainable_params = int(count_parameters(pipeline, trainable=True))
        model_footprint_mb = _measure_module_footprint_mb(pipeline)
        prototype_footprint_mb = float(num_classes_val * feat_dim * 4) / (1024.0 ** 2)

        rows.append(
            {
                "variant": variant_key,
                "label": label,
                "branches": int(branches),
                "feat_dim": int(feat_dim),
                "batch_size": batch_size_val,
                "num_classes": num_classes_val,
                "ms_per_batch_mean": ms_batch_mean,
                "ms_per_batch_std": ms_batch_std,
                "ms_per_image_mean": ms_img_mean,
                "ms_per_image_std": ms_img_std,
                "images_per_sec": images_per_sec,
                "encoder_params": encoder_params,
                "classifier_params": classifier_params,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_footprint_mb": model_footprint_mb,
                "prototype_footprint_mb": prototype_footprint_mb,
                "peak_memory_mb": peak_memory_mb,
                "memory_backend": memory_backend,
            }
        )

    baseline = rows[0]
    for row in rows:
        row["latency_ratio_vs_simplecil"] = row["ms_per_batch_mean"] / max(baseline["ms_per_batch_mean"], 1e-12)
        row["throughput_ratio_vs_simplecil"] = row["images_per_sec"] / max(baseline["images_per_sec"], 1e-12)
        row["params_ratio_vs_simplecil"] = float(row["total_params"]) / max(float(baseline["total_params"]), 1.0)
        row["memory_ratio_vs_simplecil"] = (
            row["peak_memory_mb"] / baseline["peak_memory_mb"] if baseline["peak_memory_mb"] > 0 else 1.0
        )
        row["prototype_ratio_vs_simplecil"] = (
            row["prototype_footprint_mb"] / max(baseline["prototype_footprint_mb"], 1e-12)
        )

    variants_by_name = {row["variant"]: row for row in rows}
    return {
        "device": str(resolved_device),
        "memory_backend": memory_backend,
        "checkpoint_mode": str(checkpoint_mode),
        "batch_size": batch_size_val,
        "input_shape": list(shape_tuple),
        "num_classes": num_classes_val,
        "pca_dim": pca_dim_val,
        "warmup_iters": warmup_val,
        "timed_iters": timed_val,
        "frozen_dim": frozen_dim,
        "adapted_dim": adapted_dim,
        "merged_dim": merged_dim,
        "variants": rows,
        "variants_by_name": variants_by_name,
        "summary": {
            "simplecil_ms_per_batch": variants_by_name["simplecil"]["ms_per_batch_mean"],
            "aper_full_ms_per_batch": variants_by_name["aper_full"]["ms_per_batch_mean"],
            "aper_pca_ms_per_batch": variants_by_name["aper_pca"]["ms_per_batch_mean"],
            "aper_full_latency_ratio": variants_by_name["aper_full"]["latency_ratio_vs_simplecil"],
            "aper_pca_latency_ratio": variants_by_name["aper_pca"]["latency_ratio_vs_simplecil"],
            "aper_full_params_ratio": variants_by_name["aper_full"]["params_ratio_vs_simplecil"],
            "aper_pca_params_ratio": variants_by_name["aper_pca"]["params_ratio_vs_simplecil"],
            "aper_full_peak_memory_mb": variants_by_name["aper_full"]["peak_memory_mb"],
            "simplecil_peak_memory_mb": variants_by_name["simplecil"]["peak_memory_mb"],
        },
    }


def _json_default(obj: Any) -> Any:
    """Serialize NumPy and Path objects cleanly for JSON output."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def _norm_imbalance_rows(norm_audits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten norm imbalance summary stats into tabular rows for CSV/W&B."""
    rows: List[Dict[str, Any]] = []
    for audit in norm_audits:
        dataset = audit.get("dataset", "unknown")
        stats = audit.get("norm_stats", {})
        for split in ("train", "test", "prototypes"):
            sp = stats.get(split, {})
            if not sp:
                continue
            count_val = sp.get("num_classes") if split == "prototypes" else sp.get("n_samples", 0)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "count": int(count_val or 0),
                    "adapted_norm_mean": sp["adapted_norm"]["mean"],
                    "adapted_norm_std": sp["adapted_norm"]["std"],
                    "adapted_norm_median": sp["adapted_norm"]["median"],
                    "frozen_norm_mean": sp["frozen_norm"]["mean"],
                    "frozen_norm_std": sp["frozen_norm"]["std"],
                    "frozen_norm_median": sp["frozen_norm"]["median"],
                    "norm_ratio_mean": sp["norm_ratio"]["mean"],
                    "norm_ratio_std": sp["norm_ratio"]["std"],
                    "norm_ratio_median": sp["norm_ratio"]["median"],
                    "norm_ratio_p05": sp["norm_ratio"]["p05"],
                    "norm_ratio_p95": sp["norm_ratio"]["p95"],
                    "alpha_eff_mean": sp["alpha_eff"]["mean"],
                    "alpha_eff_std": sp["alpha_eff"]["std"],
                    "alpha_eff_median": sp["alpha_eff"]["median"],
                    "alpha_eff_p05": sp["alpha_eff"]["p05"],
                    "alpha_eff_p95": sp["alpha_eff"]["p95"],
                    "zero_adapted_count": int(sp.get("zero_adapted_count", 0)),
                    "zero_frozen_count": int(sp.get("zero_frozen_count", 0)),
                }
            )
    return rows


def _norm_merge_rows(norm_audits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten unnormalized vs normalized merge comparison rows for CSV/W&B."""
    rows: List[Dict[str, Any]] = []
    for audit in norm_audits:
        dataset = audit.get("dataset", "unknown")
        for var in audit.get("variants", []):
            rows.append(
                {
                    "dataset": dataset,
                    "variant": var["variant"],
                    "label": var["label"],
                    "normalized": bool(var["normalized"]),
                    "alpha_config": var["alpha_config"],
                    "effective_alpha": var["effective_alpha"],
                    "avg_acc": var["avg_acc"],
                    "last_acc": var["last_acc"],
                    "old_acc": var["old_acc"],
                    "new_acc": var["new_acc"],
                    "delta_avg_acc_vs_unnorm": var["delta_avg_acc_vs_unnorm"],
                    "delta_last_acc_vs_unnorm": var["delta_last_acc_vs_unnorm"],
                    "delta_old_acc_vs_unnorm": var["delta_old_acc_vs_unnorm"],
                    "delta_new_acc_vs_unnorm": var["delta_new_acc_vs_unnorm"],
                }
            )
    return rows


def export_critique_csvs(
    output_dir: str,
    norm_audits: Optional[Sequence[Dict[str, Any]]] = None,
    latency_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write structured CSV tables for norm audits and/or latency benchmarks."""
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    csv_paths: Dict[str, str] = {}

    if norm_audits:
        imbalance_rows = _norm_imbalance_rows(norm_audits)
        imbalance_path = os.path.join(abs_dir, "norm_imbalance_summary.csv")
        imbalance_cols = [
            "dataset",
            "split",
            "count",
            "adapted_norm_mean",
            "adapted_norm_std",
            "adapted_norm_median",
            "frozen_norm_mean",
            "frozen_norm_std",
            "frozen_norm_median",
            "norm_ratio_mean",
            "norm_ratio_std",
            "norm_ratio_median",
            "norm_ratio_p05",
            "norm_ratio_p95",
            "alpha_eff_mean",
            "alpha_eff_std",
            "alpha_eff_median",
            "alpha_eff_p05",
            "alpha_eff_p95",
            "zero_adapted_count",
            "zero_frozen_count",
        ]
        with open(imbalance_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=imbalance_cols)
            writer.writeheader()
            writer.writerows(imbalance_rows)
        csv_paths["norm_imbalance_summary"] = imbalance_path

        merge_rows = _norm_merge_rows(norm_audits)
        merge_path = os.path.join(abs_dir, "norm_merge_comparison.csv")
        merge_cols = [
            "dataset",
            "variant",
            "label",
            "normalized",
            "alpha_config",
            "effective_alpha",
            "avg_acc",
            "last_acc",
            "old_acc",
            "new_acc",
            "delta_avg_acc_vs_unnorm",
            "delta_last_acc_vs_unnorm",
            "delta_old_acc_vs_unnorm",
            "delta_new_acc_vs_unnorm",
        ]
        with open(merge_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=merge_cols)
            writer.writeheader()
            writer.writerows(merge_rows)
        csv_paths["norm_merge_comparison"] = merge_path

    if latency_result and latency_result.get("variants"):
        latency_path = os.path.join(abs_dir, "inference_latency_benchmark.csv")
        latency_cols = [
            "variant",
            "label",
            "branches",
            "feat_dim",
            "batch_size",
            "num_classes",
            "ms_per_batch_mean",
            "ms_per_batch_std",
            "ms_per_image_mean",
            "ms_per_image_std",
            "images_per_sec",
            "encoder_params",
            "classifier_params",
            "total_params",
            "trainable_params",
            "model_footprint_mb",
            "prototype_footprint_mb",
            "peak_memory_mb",
            "memory_backend",
            "latency_ratio_vs_simplecil",
            "throughput_ratio_vs_simplecil",
            "params_ratio_vs_simplecil",
            "memory_ratio_vs_simplecil",
            "prototype_ratio_vs_simplecil",
        ]
        with open(latency_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=latency_cols)
            writer.writeheader()
            for row in latency_result["variants"]:
                writer.writerow({col: row.get(col) for col in latency_cols})
        csv_paths["inference_latency_benchmark"] = latency_path

    return csv_paths


def export_critique_json(
    output_dir: str,
    norm_audits: Optional[Sequence[Dict[str, Any]]] = None,
    latency_result: Optional[Dict[str, Any]] = None,
    filename: str = "critique_audit_summary.json",
) -> str:
    """Write combined critique micro-experiment results to a JSON summary file."""
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    json_path = os.path.join(abs_dir, filename)
    payload = {
        "norm_audits": list(norm_audits) if norm_audits else [],
        "latency_benchmark": latency_result,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return json_path


def _render_critique_plots_matplotlib(
    output_dir: str,
    norm_audits: Optional[Sequence[Dict[str, Any]]],
    latency_result: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Render critique audit summary charts using headless matplotlib (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: Dict[str, str] = {}

    if norm_audits:
        n_datasets = len(norm_audits)
        fig, axes = plt.subplots(1, 2, figsize=(max(11, 5.5 * n_datasets), 4.8))
        ax_left, ax_right = axes[0], axes[1]

        datasets = [str(a.get("dataset", f"ds_{idx}")) for idx, a in enumerate(norm_audits)]
        x = np.arange(n_datasets)
        width = 0.25

        train_alphas = [a["norm_stats"]["train"]["alpha_eff"]["mean"] for a in norm_audits]
        test_alphas = [a["norm_stats"]["test"]["alpha_eff"]["mean"] for a in norm_audits]
        proto_alphas = [a["norm_stats"]["prototypes"]["alpha_eff"]["mean"] for a in norm_audits]

        ax_left.bar(x - width, train_alphas, width, label="Train alpha_eff", color="#1f77b4")
        ax_left.bar(x, test_alphas, width, label="Test alpha_eff", color="#ff7f0e")
        ax_left.bar(x + width, proto_alphas, width, label="Proto alpha_eff", color="#2ca02c")
        ax_left.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.2, label="Balanced alpha=0.5")
        ax_left.set_ylim(0.0, 1.05)
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(datasets)
        ax_left.set_ylabel("Effective Mixing Weight alpha_eff")
        ax_left.set_title("Implicit Branch Dominance (Unnormalized Concat)")
        ax_left.legend(fontsize=8)
        ax_left.grid(axis="y", alpha=0.3)

        variant_order = ["unnormalized_concat", "normalized_alpha_0.5", "normalized_best_alpha"]
        variant_labels = ["Unnorm Concat", "Norm alpha=0.5", "Norm Best alpha"]
        colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]
        for v_idx, (v_key, v_lbl, col) in enumerate(zip(variant_order, variant_labels, colors)):
            vals = [a["variants_by_name"][v_key]["avg_acc"] for a in norm_audits]
            ax_right.bar(x + (v_idx - 1) * width, vals, width, label=v_lbl, color=col)
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(datasets)
        ax_right.set_ylabel("Avg Incremental Accuracy (%)")
        ax_right.set_title("Unnormalized vs. Normalized Merge Accuracy")
        ax_right.legend(fontsize=8)
        ax_right.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        norm_plot_path = os.path.join(output_dir, "norm_imbalance_audit.png")
        fig.savefig(norm_plot_path, dpi=200)
        plt.close(fig)
        plot_paths["norm_imbalance_audit"] = norm_plot_path

    if latency_result and latency_result.get("variants"):
        variants = latency_result["variants"]
        labels = [v["label"] for v in variants]
        latencies = [v["ms_per_batch_mean"] for v in variants]
        memories = [v["peak_memory_mb"] for v in variants]
        params_m = [v["total_params"] / 1e6 for v in variants]

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
        x = np.arange(len(variants))
        palette = ["#1f77b4", "#d62728", "#2ca02c"]

        axes[0].bar(x, latencies, color=palette)
        axes[0].set_title("Forward Latency (ms / batch)")
        axes[0].set_ylabel("ms / batch")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(x, memories, color=palette)
        axes[1].set_title(f"Peak Memory ({latency_result.get('memory_backend', 'MB')})")
        axes[1].set_ylabel("MB")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)

        axes[2].bar(x, params_m, color=palette)
        axes[2].set_title("Total Inference Parameters (M)")
        axes[2].set_ylabel("Million Params")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        axes[2].grid(axis="y", alpha=0.3)

        fig.tight_layout()
        lat_plot_path = os.path.join(output_dir, "inference_latency_memory.png")
        fig.savefig(lat_plot_path, dpi=200)
        plt.close(fig)
        plot_paths["inference_latency_memory"] = lat_plot_path

    return plot_paths


def _render_critique_plots_pillow(
    output_dir: str,
    norm_audits: Optional[Sequence[Dict[str, Any]]],
    latency_result: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Headless Pillow fallback renderer when matplotlib is not installed locally."""
    from PIL import Image, ImageDraw

    def _draw_bar_panel(
        draw: ImageDraw.ImageDraw,
        box: Tuple[int, int, int, int],
        title: str,
        labels: List[str],
        values: List[float],
        colors: List[Tuple[int, int, int]],
        y_max_override: Optional[float] = None,
    ) -> None:
        x0, y0, x1, y1 = box
        draw.text((x0, y0 - 24), title, fill=(20, 20, 20))
        draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80), width=2)
        if not values:
            return
        max_val = y_max_override if y_max_override is not None else max(max(values) * 1.15, 1e-6)
        n = len(values)
        slot_w = max((x1 - x0 - 20) // max(n, 1), 12)
        bar_w = max(int(slot_w * 0.6), 8)
        for idx, (lbl, val) in enumerate(zip(labels, values)):
            bx0 = x0 + 14 + idx * slot_w
            bx1 = min(bx0 + bar_w, x1 - 6)
            frac = max(0.0, min(1.0, float(val) / max_val))
            bh = int(frac * (y1 - y0 - 12))
            by0 = y1 - bh
            col = colors[idx % len(colors)]
            draw.rectangle([bx0, by0, bx1, y1 - 1], fill=col)
            draw.text((bx0, max(y0 + 4, by0 - 14)), f"{val:.2f}", fill=(20, 20, 20))
            draw.text((bx0, y1 + 8), lbl[:14], fill=(40, 40, 40))

    plot_paths: Dict[str, str] = {}

    if norm_audits:
        w, h = 960, 480
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        first = norm_audits[0]
        ds_name = str(first.get("dataset", "dataset"))
        alpha_labels = [f"{a.get('dataset', 'ds')}:test" for a in norm_audits]
        alpha_vals = [float(a["norm_stats"]["test"]["alpha_eff"]["mean"]) for a in norm_audits]
        _draw_bar_panel(
            draw,
            (55, 55, 445, 400),
            f"Implicit alpha_eff ({ds_name})",
            alpha_labels,
            alpha_vals,
            [(31, 119, 180), (255, 127, 14), (44, 160, 44)],
            y_max_override=1.05,
        )

        var_keys = ["unnormalized_concat", "normalized_alpha_0.5", "normalized_best_alpha"]
        var_lbls = ["Unnorm", "Norm 0.5", "Norm Best"]
        var_vals = [float(first["variants_by_name"][k]["avg_acc"]) for k in var_keys]
        _draw_bar_panel(
            draw,
            (515, 55, 915, 400),
            f"Merge Avg Acc (%) - {ds_name}",
            var_lbls,
            var_vals,
            [(127, 127, 127), (31, 119, 180), (44, 160, 44)],
            y_max_override=105.0,
        )

        norm_plot_path = os.path.join(output_dir, "norm_imbalance_audit.png")
        img.save(norm_plot_path, format="PNG")
        plot_paths["norm_imbalance_audit"] = norm_plot_path

    if latency_result and latency_result.get("variants"):
        variants = latency_result["variants"]
        w, h = 1080, 460
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        short_labels = [v["variant"] for v in variants]
        palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44)]

        _draw_bar_panel(
            draw,
            (45, 55, 345, 385),
            "Latency (ms / batch)",
            short_labels,
            [float(v["ms_per_batch_mean"]) for v in variants],
            palette,
        )
        _draw_bar_panel(
            draw,
            (395, 55, 695, 385),
            f"Peak Memory ({latency_result.get('memory_backend', 'MB')})",
            short_labels,
            [float(v["peak_memory_mb"]) for v in variants],
            palette,
        )
        _draw_bar_panel(
            draw,
            (745, 55, 1045, 385),
            "Params (Millions)",
            short_labels,
            [float(v["total_params"]) / 1e6 for v in variants],
            palette,
        )

        lat_plot_path = os.path.join(output_dir, "inference_latency_memory.png")
        img.save(lat_plot_path, format="PNG")
        plot_paths["inference_latency_memory"] = lat_plot_path

    return plot_paths


def generate_critique_plots(
    output_dir: str,
    norm_audits: Optional[Sequence[Dict[str, Any]]] = None,
    latency_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate summary PNG figures (`norm_imbalance_audit.png`, `inference_latency_memory.png`).
    Uses headless matplotlib when available and falls back to Pillow otherwise.
    """
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    if not norm_audits and not (latency_result and latency_result.get("variants")):
        return {}

    try:
        import matplotlib  # noqa: F401

        return _render_critique_plots_matplotlib(abs_dir, norm_audits, latency_result)
    except ImportError:
        logging.warning("matplotlib is not installed; falling back to headless Pillow chart renderer.")
        return _render_critique_plots_pillow(abs_dir, norm_audits, latency_result)


def log_critique_to_wandb(
    args: Dict[str, Any],
    norm_audits: Optional[Sequence[Dict[str, Any]]] = None,
    latency_result: Optional[Dict[str, Any]] = None,
    plot_paths: Optional[Dict[str, str]] = None,
    csv_paths: Optional[Dict[str, str]] = None,
    json_path: Optional[str] = None,
):
    """
    Optionally log norm imbalance tables, inference benchmark metrics, and figures to W&B
    (`nnm-hcmus/Group7-ML-Lab`) with mandatory `["empirics", "critique-audit"]` tags.
    """
    if not _is_wandb_enabled(args):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B tracking is enabled, but `wandb` is not installed. "
            "Please install it with `pip install wandb`."
        ) from exc

    merged_args = copy.deepcopy(args)
    user_tags = merged_args.get("wandb_tags")
    combined_tags: List[str] = list(DEFAULT_CRITIQUE_WANDB_TAGS)
    if isinstance(user_tags, str):
        for tag in user_tags.replace(",", " ").split():
            if tag.strip() and tag.strip() not in combined_tags:
                combined_tags.append(tag.strip())
    elif isinstance(user_tags, (list, tuple, set)):
        for tag in user_tags:
            t_str = str(tag).strip()
            if t_str and t_str not in combined_tags:
                combined_tags.append(t_str)
    merged_args["wandb_tags"] = combined_tags
    tags = _resolve_wandb_tags(merged_args)

    entity = args.get("wandb_entity") or "nnm-hcmus"
    project = args.get("wandb_project") or "Group7-ML-Lab"
    mode = str(args.get("mode") or "all")
    datasets = [str(a.get("dataset", "unknown")) for a in (norm_audits or [])]
    ds_label = "_".join(datasets) if datasets else str(args.get("dataset") or "backbone")
    group = args.get("wandb_group") or f"empirics_critique_{mode}"
    name = args.get("wandb_name") or f"critique_{mode}_{ds_label}"

    config: Dict[str, Any] = {}
    for key, value in merged_args.items():
        if key in ("convnet_type", "wandb_tags"):
            continue
        config[key] = value
    backbone_type = args.get("backbone_type") or args.get("convnet_type")
    if backbone_type:
        config["backbone_type"] = backbone_type
    config["datasets"] = datasets

    run = None
    try:
        run = wandb.init(
            entity=entity,
            project=project,
            group=group,
            name=name,
            tags=tags,
            config=config,
        )

        log_payload: Dict[str, Any] = {}
        if norm_audits:
            imb_rows = _norm_imbalance_rows(norm_audits)
            if imb_rows:
                imb_cols = list(imb_rows[0].keys())
                log_payload["critique/norm_imbalance_table"] = wandb.Table(
                    columns=imb_cols,
                    data=[[r.get(c) for c in imb_cols] for r in imb_rows],
                )
            merge_rows = _norm_merge_rows(norm_audits)
            if merge_rows:
                merge_cols = list(merge_rows[0].keys())
                log_payload["critique/norm_merge_table"] = wandb.Table(
                    columns=merge_cols,
                    data=[[r.get(c) for c in merge_cols] for r in merge_rows],
                )

        if latency_result and latency_result.get("variants"):
            lat_rows = latency_result["variants"]
            lat_cols = list(lat_rows[0].keys())
            log_payload["critique/inference_latency_table"] = wandb.Table(
                columns=lat_cols,
                data=[[r.get(c) for c in lat_cols] for r in lat_rows],
            )

        if plot_paths:
            for key, path in plot_paths.items():
                if path and os.path.exists(path):
                    log_payload[f"critique/{key}"] = wandb.Image(path)

        if log_payload:
            if hasattr(wandb, "log"):
                wandb.log(log_payload)
            elif run is not None and hasattr(run, "log"):
                run.log(log_payload)

        summary_target = getattr(wandb, "run", None) or run
        if summary_target is not None and getattr(summary_target, "summary", None) is not None:
            if norm_audits:
                first_summary = norm_audits[0].get("summary", {})
                for key, val in first_summary.items():
                    summary_target.summary[f"norm_audit/{key}"] = val
            if latency_result:
                for key, val in (latency_result.get("summary") or {}).items():
                    summary_target.summary[f"latency_bench/{key}"] = val
            if csv_paths:
                summary_target.summary["csv_paths"] = dict(csv_paths)
            if json_path:
                summary_target.summary["json_path"] = json_path

        return run
    finally:
        if run is not None:
            if hasattr(wandb, "finish"):
                wandb.finish()
            elif hasattr(run, "finish"):
                run.finish()
