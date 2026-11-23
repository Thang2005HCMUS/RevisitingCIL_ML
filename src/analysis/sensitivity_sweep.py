import csv
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from framework.learner import _is_wandb_enabled, _resolve_method_name, _resolve_wandb_tags
from models.merged_encoder import _extract_features, _validate_alpha
from utils.metrics import count_parameters

DEFAULT_ALPHA_GRID: List[float] = [round(0.1 * i, 1) for i in range(11)]
DEFAULT_K_GRID: List[int] = [16, 32, 64, 128, 256, 512, 768, 1536]
DEFAULT_RANDOM_SEEDS: List[int] = [1993, 1994, 1995]

CACHE_COMPAT_KEYS: Tuple[str, ...] = (
    "dataset",
    "model_name",
    "backbone_type",
    "seed",
    "init_cls",
    "increment",
    "tuned_epoch",
    "init_lr",
    "batch_size",
    "weight_decay",
    "ffn_num",
    "optimizer",
    "adapt_stages",
)
_CACHE_FLOAT_KEYS = {"init_lr", "weight_decay"}
_CACHE_INT_KEYS = {"seed", "init_cls", "increment", "tuned_epoch", "batch_size", "ffn_num", "adapt_stages"}


def _unwrap_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def check_cache_compatibility(
    cached_data: Dict[str, Any],
    run_args: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Compare cached feature metadata against requested run_args to guard against stale caches.
    Optional metadata fields missing in legacy/synthetic caches are skipped to stay backward compatible.
    """
    if not run_args:
        return True, []

    cached_meta = cached_data.get("metadata", {}) or {}
    expected_map: Dict[str, Any] = {
        "dataset": run_args.get("dataset"),
        "model_name": run_args.get("model_name"),
        "backbone_type": run_args.get("backbone_type") or run_args.get("convnet_type"),
        "seed": _unwrap_scalar(run_args.get("seed")),
        "init_cls": run_args.get("init_cls"),
        "increment": run_args.get("increment"),
        "tuned_epoch": run_args.get("tuned_epoch"),
        "init_lr": run_args.get("init_lr"),
        "batch_size": run_args.get("batch_size"),
        "weight_decay": run_args.get("weight_decay"),
        "ffn_num": run_args.get("ffn_num"),
        "optimizer": run_args.get("optimizer"),
        "adapt_stages": 1 if run_args.get("adapt_stages") is None else run_args.get("adapt_stages"),
    }

    mismatches: List[str] = []
    for key in CACHE_COMPAT_KEYS:
        cached_val = _unwrap_scalar(cached_meta.get(key))
        expected_val = expected_map.get(key)
        if cached_val is None or expected_val is None:
            continue

        if key in _CACHE_FLOAT_KEYS:
            if not math.isclose(float(cached_val), float(expected_val), rel_tol=1e-9, abs_tol=1e-12):
                mismatches.append(f"{key}: cached={cached_val} != requested={expected_val}")
        elif key in _CACHE_INT_KEYS:
            if int(cached_val) != int(expected_val):
                mismatches.append(f"{key}: cached={cached_val} != requested={expected_val}")
        else:
            if str(cached_val).strip().lower() != str(expected_val).strip().lower():
                mismatches.append(f"{key}: cached={cached_val} != requested={expected_val}")

    return len(mismatches) == 0, mismatches


def save_feature_cache(cache_data: Dict[str, Any], cache_path: str) -> str:
    """Save extracted dual-branch feature cache to disk via torch.save."""
    abs_path = os.path.abspath(cache_path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(cache_data, abs_path)
    return abs_path


def load_feature_cache(cache_path: str) -> Dict[str, Any]:
    """Load cached dual-branch features from disk and normalize array dtypes."""
    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(cache_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict payload in feature cache '{cache_path}', got {type(data).__name__}.")

    required_keys = (
        "train_adapted",
        "train_frozen",
        "train_targets",
        "test_adapted",
        "test_frozen",
        "test_targets",
        "increments",
    )
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"Feature cache '{cache_path}' is missing required keys: {missing}")

    for key in ("train_adapted", "train_frozen", "test_adapted", "test_frozen"):
        arr = data[key]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        data[key] = np.asarray(arr, dtype=np.float32)

    for key in ("train_targets", "test_targets"):
        arr = data[key]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        data[key] = np.asarray(arr, dtype=np.int64)

    data["increments"] = [int(x) for x in data["increments"]]
    if "task_class_ids" not in data or not data["task_class_ids"]:
        task_class_ids = []
        cursor = 0
        for inc in data["increments"]:
            task_class_ids.append(list(range(cursor, cursor + inc)))
            cursor += inc
        data["task_class_ids"] = task_class_ids
    else:
        data["task_class_ids"] = [[int(c) for c in cids] for cids in data["task_class_ids"]]

    if "stage0_classes" not in data:
        data["stage0_classes"] = list(data["task_class_ids"][0]) if data["task_class_ids"] else []
    else:
        data["stage0_classes"] = [int(c) for c in data["stage0_classes"]]

    feat_dim = int(data["train_adapted"].shape[1])
    data["feature_dim"] = feat_dim
    data["merged_dim"] = feat_dim * 2
    data.setdefault("metadata", {})
    return data


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, Any]:
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 3:
            return batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1]
    raise ValueError(f"Unexpected DataLoader batch format: {type(batch).__name__}")


def _extract_loader_pair_features(
    adapted_ptm: torch.nn.Module,
    frozen_ptm: torch.nn.Module,
    loader: Any,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single no-grad pass over a DataLoader and collect (phi*, phi, targets)."""
    adapted_ptm.eval()
    frozen_ptm.eval()
    adapted_ptm.to(device)
    frozen_ptm.to(device)

    adapted_chunks: List[np.ndarray] = []
    frozen_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs, targets = _unpack_batch(batch)
            inputs = inputs.to(device)
            feat_adapted = _extract_features(adapted_ptm, inputs).detach().cpu().numpy().astype(np.float32)
            feat_frozen = _extract_features(frozen_ptm, inputs).detach().cpu().numpy().astype(np.float32)
            if isinstance(targets, torch.Tensor):
                targets_np = targets.detach().cpu().numpy().astype(np.int64)
            else:
                targets_np = np.asarray(targets, dtype=np.int64)

            adapted_chunks.append(feat_adapted)
            frozen_chunks.append(feat_frozen)
            target_chunks.append(targets_np)

    if not adapted_chunks:
        out_dim = int(getattr(adapted_ptm, "out_dim", 0))
        return (
            np.empty((0, out_dim), dtype=np.float32),
            np.empty((0, out_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.concatenate(adapted_chunks, axis=0),
        np.concatenate(frozen_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
    )


def extract_and_cache_features(
    learner: Any,
    task_manager: Any,
    cache_path: Optional[str] = None,
    force_reextract: bool = False,
) -> Dict[str, Any]:
    """
    Run Stage 0 adaptation once, extract paired train/test branch features (phi*(x), phi(x))
    across all continual tasks, and optionally cache them to disk.
    Re-extracts automatically when existing cache metadata conflicts with learner.args.
    """
    if cache_path and os.path.exists(cache_path) and not force_reextract:
        cached_candidate = load_feature_cache(cache_path)
        learner_args = getattr(learner, "args", None) if learner is not None else None
        compatible, mismatches = check_cache_compatibility(cached_candidate, learner_args)
        if compatible:
            logging.info(f"Loading cached Exp 3 features from {cache_path}")
            return cached_candidate
        logging.warning(
            f"Existing Exp 3 feature cache at {cache_path} is incompatible "
            f"({'; '.join(mismatches)}); re-extracting features."
        )

    if learner is None or task_manager is None:
        raise ValueError("learner and task_manager are required when feature cache is not present.")
    if getattr(learner, "adapted_ptm", None) is None or getattr(learner, "frozen_ptm", None) is None:
        raise ValueError("learner must provide both adapted_ptm and frozen_ptm for dual-branch feature caching.")

    nb_tasks = int(task_manager.nb_tasks)
    if nb_tasks <= 0:
        raise ValueError(f"task_manager.nb_tasks must be > 0, got {nb_tasks}")

    args = getattr(learner, "args", {}) or {}
    device = getattr(learner, "device", torch.device("cpu"))
    total_params = count_parameters(learner.adapted_ptm, trainable=False)
    trainable_params = count_parameters(learner.adapted_ptm, trainable=True)

    # Task 0: adapt once (unless adapt_stages == 0)
    train_loader_0, test_loader_0, proto_loader_0 = task_manager.get_task_loaders(0)
    class_ids_0 = [int(c) for c in task_manager.get_class_ids(0)]

    adapt_stages_raw = args.get("adapt_stages", 1)
    adapt_stages = 1 if adapt_stages_raw is None else int(adapt_stages_raw)
    if adapt_stages < 0:
        raise ValueError(f"adapt_stages must be >= 0, got {adapt_stages}")
    if adapt_stages > 0:
        learner._adapt_model(train_loader_0, nb_old=0, nb_new=len(class_ids_0), task_id=0)

    # Freeze both branches permanently after Stage 0 adaptation
    learner.adapted_ptm.eval()
    for param in learner.adapted_ptm.parameters():
        param.requires_grad = False
    learner.frozen_ptm.eval()
    for param in learner.frozen_ptm.parameters():
        param.requires_grad = False

    train_adapted_parts: List[np.ndarray] = []
    train_frozen_parts: List[np.ndarray] = []
    train_target_parts: List[np.ndarray] = []
    task_class_ids: List[List[int]] = [class_ids_0]
    test_loaders: List[Tuple[List[int], Any]] = [(class_ids_0, test_loader_0)]

    tr_ad_0, tr_fr_0, tr_y_0 = _extract_loader_pair_features(
        learner.adapted_ptm, learner.frozen_ptm, proto_loader_0, device
    )
    train_adapted_parts.append(tr_ad_0)
    train_frozen_parts.append(tr_fr_0)
    train_target_parts.append(tr_y_0)

    for task_id in range(1, nb_tasks):
        _, test_loader_b, proto_loader_b = task_manager.get_task_loaders(task_id)
        class_ids_b = [int(c) for c in task_manager.get_class_ids(task_id)]
        task_class_ids.append(class_ids_b)
        test_loaders.append((class_ids_b, test_loader_b))

        tr_ad_b, tr_fr_b, tr_y_b = _extract_loader_pair_features(
            learner.adapted_ptm, learner.frozen_ptm, proto_loader_b, device
        )
        train_adapted_parts.append(tr_ad_b)
        train_frozen_parts.append(tr_fr_b)
        train_target_parts.append(tr_y_b)

    # Extract test features from final cumulative test loader; backfill if a loader wasn't cumulative.
    test_ad_last, test_fr_last, test_y_last = _extract_loader_pair_features(
        learner.adapted_ptm, learner.frozen_ptm, test_loaders[-1][1], device
    )
    test_adapted_parts = [test_ad_last]
    test_frozen_parts = [test_fr_last]
    test_target_parts = [test_y_last]

    all_classes = {cid for cids in task_class_ids for cid in cids}
    covered_classes = set(np.unique(test_y_last).tolist()) if len(test_y_last) > 0 else set()
    missing_classes = all_classes - covered_classes
    if missing_classes:
        for _, loader_b in test_loaders[:-1]:
            if not missing_classes:
                break
            te_ad_b, te_fr_b, te_y_b = _extract_loader_pair_features(
                learner.adapted_ptm, learner.frozen_ptm, loader_b, device
            )
            if len(te_y_b) == 0:
                continue
            keep_mask = np.isin(te_y_b, list(missing_classes))
            if np.any(keep_mask):
                test_adapted_parts.append(te_ad_b[keep_mask])
                test_frozen_parts.append(te_fr_b[keep_mask])
                test_target_parts.append(te_y_b[keep_mask])
                covered_classes.update(np.unique(te_y_b[keep_mask]).tolist())
                missing_classes = all_classes - covered_classes

    train_adapted = np.concatenate(train_adapted_parts, axis=0).astype(np.float32)
    train_frozen = np.concatenate(train_frozen_parts, axis=0).astype(np.float32)
    train_targets = np.concatenate(train_target_parts, axis=0).astype(np.int64)
    test_adapted = np.concatenate(test_adapted_parts, axis=0).astype(np.float32)
    test_frozen = np.concatenate(test_frozen_parts, axis=0).astype(np.float32)
    test_targets = np.concatenate(test_target_parts, axis=0).astype(np.int64)

    feat_dim = int(train_adapted.shape[1])
    increments = [len(cids) for cids in task_class_ids]
    backbone_type = args.get("backbone_type") or args.get("convnet_type") or "unknown"

    cache_data: Dict[str, Any] = {
        "train_adapted": train_adapted,
        "train_frozen": train_frozen,
        "train_targets": train_targets,
        "test_adapted": test_adapted,
        "test_frozen": test_frozen,
        "test_targets": test_targets,
        "increments": increments,
        "task_class_ids": task_class_ids,
        "stage0_classes": list(task_class_ids[0]),
        "feature_dim": feat_dim,
        "merged_dim": feat_dim * 2,
        "metadata": {
            "dataset": args.get("dataset", "unknown"),
            "model_name": args.get("model_name", "aper_adapter"),
            "method_name": _resolve_method_name(args) if args else "aper_adapter",
            "backbone_type": backbone_type,
            "seed": args.get("seed", 1993),
            "init_cls": args.get("init_cls", increments[0]),
            "increment": args.get("increment", increments[1] if len(increments) > 1 else increments[0]),
            "tuned_epoch": args.get("tuned_epoch"),
            "init_lr": args.get("init_lr"),
            "batch_size": args.get("batch_size"),
            "weight_decay": args.get("weight_decay"),
            "ffn_num": args.get("ffn_num"),
            "optimizer": args.get("optimizer"),
            "adapt_stages": int(adapt_stages),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
        },
    }

    if cache_path:
        saved_to = save_feature_cache(cache_data, cache_path)
        logging.info(f"Saved Exp 3 feature cache to {saved_to}")

    return cache_data


def _l2_normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safely L2-normalize each row vector while avoiding NaN on zero-norm rows."""
    arr_f = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr_f, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr_f / norms


def build_weighted_features(adapted_norm: np.ndarray, frozen_norm: np.ndarray, alpha: float) -> np.ndarray:
    """Form weighted merged embedding [sqrt(alpha)*phi_hat*, sqrt(1-alpha)*phi_hat]."""
    alpha_val = _validate_alpha(alpha)
    adapted_arr = np.asarray(adapted_norm, dtype=np.float32)
    frozen_arr = np.asarray(frozen_norm, dtype=np.float32)
    if adapted_arr.shape != frozen_arr.shape:
        raise ValueError(
            f"Branch feature shapes must match, got {adapted_arr.shape} and {frozen_arr.shape}."
        )
    w_adapted = math.sqrt(alpha_val) * adapted_arr
    w_frozen = math.sqrt(1.0 - alpha_val) * frozen_arr
    return np.concatenate([w_adapted, w_frozen], axis=-1).astype(np.float32, copy=False)


def fit_stage1_pca(
    X_stage0: np.ndarray,
    max_components: int,
    prefer_sklearn: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Fit orthogonal PCA exclusively on Stage 1 features.
    Prefers `sklearn.decomposition.PCA` and falls back to exact centered NumPy SVD.
    """
    X_arr = np.asarray(X_stage0, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError(f"X_stage0 must be a non-empty 2D array, got shape {X_arr.shape}.")

    n_comp = min(int(max_components), X_arr.shape[0], X_arr.shape[1])
    if n_comp <= 0:
        raise ValueError(f"max_components must be positive, got {max_components}.")

    if prefer_sklearn:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            PCA = None
        if PCA is not None:
            pca = PCA(n_components=n_comp, svd_solver="full")
            pca.fit(X_arr)
            return (
                np.asarray(pca.mean_, dtype=np.float32),
                np.asarray(pca.components_, dtype=np.float32),
                np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
                "sklearn",
            )

    mean = np.mean(X_arr, axis=0, dtype=np.float64)
    centered = X_arr - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)

    # Deterministic sign convention matching sklearn svd_flip(u_based_decision=False).
    max_abs_rows = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), max_abs_rows])
    signs[signs == 0] = 1.0
    vt = vt * signs[:, np.newaxis]

    var_sq = s ** 2
    total_var = float(np.sum(var_sq))
    if total_var > 0.0:
        evr = var_sq / total_var
    else:
        evr = np.zeros_like(var_sq)

    return (
        mean.astype(np.float32),
        vt[:n_comp].astype(np.float32),
        evr[:n_comp].astype(np.float32),
        "numpy_svd",
    )


def evaluate_cil_trajectory(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    increments: Sequence[int],
    task_class_ids: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate cosine prototype CIL trajectory across all tasks in a single vectorized pass.
    Returns avg_acc, last_acc, old_acc, new_acc, and per-task accuracy curves.
    """
    increments_list = [int(x) for x in increments]
    if not increments_list:
        raise ValueError("increments must be non-empty.")

    if task_class_ids is None:
        resolved_task_ids: List[List[int]] = []
        cursor = 0
        for inc in increments_list:
            resolved_task_ids.append(list(range(cursor, cursor + inc)))
            cursor += inc
    else:
        resolved_task_ids = [[int(c) for c in cids] for cids in task_class_ids]

    ordered_classes = [c for cids in resolved_task_ids for c in cids]
    class_to_col = {cls_id: idx for idx, cls_id in enumerate(ordered_classes)}
    total_classes = len(ordered_classes)
    feat_dim = int(train_features.shape[1])

    prototypes = np.zeros((total_classes, feat_dim), dtype=np.float32)
    for col_idx, cls_id in enumerate(ordered_classes):
        cls_mask = train_targets == cls_id
        if np.any(cls_mask):
            prototypes[col_idx] = train_features[cls_mask].mean(axis=0)

    proto_norm = _l2_normalize_rows(prototypes)
    test_norm = _l2_normalize_rows(test_features)
    sim = test_norm @ proto_norm.T

    test_cols = np.array([class_to_col.get(int(t), -1) for t in test_targets], dtype=np.int64)

    top1_curve: List[float] = []
    old_curve: List[float] = []
    new_curve: List[float] = []
    nb_old = 0
    seen_count = 0

    for task_cids in resolved_task_ids:
        seen_count += len(task_cids)
        seen_mask = (test_cols >= 0) & (test_cols < seen_count)
        if not np.any(seen_mask):
            top1_curve.append(0.0)
            old_curve.append(0.0)
            new_curve.append(0.0)
            nb_old = seen_count
            continue

        preds = np.argmax(sim[seen_mask, :seen_count], axis=1)
        y_true = test_cols[seen_mask]

        top1 = round(float(np.mean(preds == y_true) * 100.0), 2)
        old_mask = y_true < nb_old
        new_mask = y_true >= nb_old
        old_val = 0.0 if not np.any(old_mask) else round(float(np.mean(preds[old_mask] == y_true[old_mask]) * 100.0), 2)
        new_val = 0.0 if not np.any(new_mask) else round(float(np.mean(preds[new_mask] == y_true[new_mask]) * 100.0), 2)

        top1_curve.append(top1)
        old_curve.append(old_val)
        new_curve.append(new_val)
        nb_old = seen_count

    avg_acc = round(float(np.mean(top1_curve)), 2)
    last_acc = round(float(top1_curve[-1]), 2)
    old_acc = round(float(np.mean(old_curve[1:])), 2) if len(old_curve) > 1 else round(float(old_curve[-1]), 2)
    new_acc = round(float(np.mean(new_curve)), 2)

    return {
        "avg_acc": avg_acc,
        "last_acc": last_acc,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "last_old_acc": round(float(old_curve[-1]), 2),
        "last_new_acc": round(float(new_curve[-1]), 2),
        "top1_curve": top1_curve,
        "old_curve": old_curve,
        "new_curve": new_curve,
    }


def _select_summary_records(rows: List[Dict[str, Any]], merged_dim: int) -> Dict[str, Any]:
    if not rows:
        return {}

    best_overall = max(rows, key=lambda r: (r["avg_acc"], r["last_acc"], -r["effective_k"]))
    max_k_in_grid = max(r["k"] for r in rows)
    target_full_k = merged_dim if any(r["k"] == merged_dim for r in rows) else max_k_in_grid
    full_dim_rows = [r for r in rows if r["k"] == target_full_k]
    best_full_dim = max(full_dim_rows, key=lambda r: (r["avg_acc"], r["last_acc"]))

    default_aper = min(rows, key=lambda r: (abs(r["k"] - merged_dim), abs(r["alpha"] - 0.5)))
    simplecil_ptm = min(rows, key=lambda r: (abs(r["k"] - merged_dim), abs(r["alpha"] - 0.0)))
    simplecil_adapted = min(rows, key=lambda r: (abs(r["k"] - merged_dim), abs(r["alpha"] - 1.0)))

    pareto_128_rows = [r for r in rows if r["effective_k"] <= 128]
    best_pareto_128 = (
        max(pareto_128_rows, key=lambda r: (r["avg_acc"], r["last_acc"], -r["effective_k"]))
        if pareto_128_rows
        else None
    )

    return {
        "best_overall": best_overall,
        "best_full_dim": best_full_dim,
        "default_aper": default_aper,
        "simplecil_ptm": simplecil_ptm,
        "simplecil_adapted": simplecil_adapted,
        "best_pareto_128": best_pareto_128,
    }


def run_2d_sensitivity_sweep(
    cached_features: Dict[str, Any],
    alpha_grid: Optional[Sequence[float]] = None,
    k_grid: Optional[Sequence[int]] = None,
    method: str = "pca",
    random_seeds: Optional[Sequence[int]] = None,
    prefer_sklearn: bool = True,
) -> Dict[str, Any]:
    """
    Execute vectorized 2D sensitivity sweep over (alpha, k) from cached dual-branch features.
    Supports Stage-1 PCA projection (method="pca"), multi-seed Random Subsampling ("random"),
    or evaluating both side-by-side ("both").
    """
    method_norm = str(method or "pca").lower()
    if method_norm not in ("pca", "random", "both"):
        raise ValueError(f"Unsupported sweep method '{method}'. Expected one of ('pca', 'random', 'both').")

    alphas = list(DEFAULT_ALPHA_GRID) if alpha_grid is None else [_validate_alpha(a) for a in alpha_grid]
    ks = list(DEFAULT_K_GRID) if k_grid is None else [int(k) for k in k_grid]
    if not alphas:
        raise ValueError("alpha_grid must not be empty.")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"k_grid must contain positive integers, got {ks}.")

    seeds = list(DEFAULT_RANDOM_SEEDS) if random_seeds is None else [int(s) for s in random_seeds]
    if method_norm in ("random", "both") and not seeds:
        raise ValueError("random_seeds must not be empty when running random subsampling.")

    train_adapted_norm = _l2_normalize_rows(cached_features["train_adapted"])
    train_frozen_norm = _l2_normalize_rows(cached_features["train_frozen"])
    test_adapted_norm = _l2_normalize_rows(cached_features["test_adapted"])
    test_frozen_norm = _l2_normalize_rows(cached_features["test_frozen"])
    train_targets = np.asarray(cached_features["train_targets"], dtype=np.int64)
    test_targets = np.asarray(cached_features["test_targets"], dtype=np.int64)
    increments = [int(x) for x in cached_features["increments"]]
    task_class_ids = cached_features.get("task_class_ids")
    stage0_classes = cached_features.get("stage0_classes") or list(range(increments[0]))

    feat_dim = int(train_adapted_norm.shape[1])
    merged_dim = int(feat_dim * 2)

    stage0_mask = np.isin(train_targets, stage0_classes)
    n_stage0 = int(np.sum(stage0_mask))
    if n_stage0 == 0:
        raise ValueError("No Stage 1 (Task 0) training samples found in cached_features.")
    max_pca_rank = min(n_stage0, merged_dim)

    run_pca = method_norm in ("pca", "both")
    run_random = method_norm in ("random", "both")
    pca_backend_used: Optional[str] = None
    warned_caps = set()
    rows: List[Dict[str, Any]] = []

    sub_ks = [k for k in ks if k < merged_dim]
    max_k_needed = min(max(sub_ks), max_pca_rank) if sub_ks else 0

    for alpha in alphas:
        train_Phi = build_weighted_features(train_adapted_norm, train_frozen_norm, alpha)
        test_Phi = build_weighted_features(test_adapted_norm, test_frozen_norm, alpha)

        pca_mean = None
        train_proj_all = None
        test_proj_all = None
        evr = np.empty((0,), dtype=np.float32)

        if run_pca and max_k_needed > 0:
            pca_mean, pca_components, evr, pca_backend_used = fit_stage1_pca(
                train_Phi[stage0_mask],
                max_components=max_k_needed,
                prefer_sklearn=prefer_sklearn,
            )
            train_proj_all = (train_Phi - pca_mean) @ pca_components.T
            test_proj_all = (test_Phi - pca_mean) @ pca_components.T

        full_traj_cache: Optional[Dict[str, Any]] = None

        for k in ks:
            pca_metrics: Optional[Dict[str, Any]] = None
            evr_sum: Optional[float] = None
            effective_k = k
            k_capped = False

            if k >= merged_dim:
                effective_k = merged_dim
                k_capped = k > merged_dim
                if k_capped:
                    logging.warning(
                        f"Requested k={k} exceeds merged_dim={merged_dim}; using full dimension {merged_dim}."
                    )
                full_traj_cache = evaluate_cil_trajectory(
                    train_Phi, train_targets, test_Phi, test_targets, increments, task_class_ids
                )
                if run_pca:
                    pca_metrics = full_traj_cache
                    evr_sum = 1.0
            else:
                if run_pca:
                    if k > max_pca_rank:
                        effective_k = max_pca_rank
                        k_capped = True
                        if k not in warned_caps:
                            warned_caps.add(k)
                            logging.warning(
                                f"Requested PCA k={k} exceeds Stage 1 rank limit {max_pca_rank} "
                                f"(n_stage0={n_stage0}, merged_dim={merged_dim}); capping effective_k={max_pca_rank}."
                            )
                    else:
                        effective_k = int(k)
                        k_capped = False

                    train_pca = train_proj_all[:, :effective_k]
                    test_pca = test_proj_all[:, :effective_k]
                    pca_metrics = evaluate_cil_trajectory(
                        train_pca, train_targets, test_pca, test_targets, increments, task_class_ids
                    )
                    evr_sum = round(float(np.sum(evr[:effective_k])), 6)

            random_avg_acc: Optional[float] = None
            random_avg_acc_std: Optional[float] = None
            random_last_acc: Optional[float] = None
            random_last_acc_std: Optional[float] = None
            random_old_acc: Optional[float] = None
            random_new_acc: Optional[float] = None
            random_top1_curve: Optional[List[float]] = None

            if run_random:
                if k >= merged_dim:
                    if full_traj_cache is None:
                        full_traj_cache = evaluate_cil_trajectory(
                            train_Phi, train_targets, test_Phi, test_targets, increments, task_class_ids
                        )
                    random_avg_acc = full_traj_cache["avg_acc"]
                    random_avg_acc_std = 0.0
                    random_last_acc = full_traj_cache["last_acc"]
                    random_last_acc_std = 0.0
                    random_old_acc = full_traj_cache["old_acc"]
                    random_new_acc = full_traj_cache["new_acc"]
                    random_top1_curve = full_traj_cache["top1_curve"]
                else:
                    seed_trajs = []
                    for seed in seeds:
                        rng = np.random.default_rng(int(seed))
                        sub_idx = np.sort(rng.choice(merged_dim, size=min(int(k), merged_dim), replace=False))
                        traj = evaluate_cil_trajectory(
                            train_Phi[:, sub_idx],
                            train_targets,
                            test_Phi[:, sub_idx],
                            test_targets,
                            increments,
                            task_class_ids,
                        )
                        seed_trajs.append(traj)

                    random_avg_acc = round(float(np.mean([t["avg_acc"] for t in seed_trajs])), 2)
                    random_avg_acc_std = round(float(np.std([t["avg_acc"] for t in seed_trajs])), 2)
                    random_last_acc = round(float(np.mean([t["last_acc"] for t in seed_trajs])), 2)
                    random_last_acc_std = round(float(np.std([t["last_acc"] for t in seed_trajs])), 2)
                    random_old_acc = round(float(np.mean([t["old_acc"] for t in seed_trajs])), 2)
                    random_new_acc = round(float(np.mean([t["new_acc"] for t in seed_trajs])), 2)
                    random_top1_curve = [
                        round(float(np.mean([t["top1_curve"][b] for t in seed_trajs])), 2)
                        for b in range(len(increments))
                    ]

            primary_metrics = pca_metrics if run_pca else {
                "avg_acc": random_avg_acc,
                "last_acc": random_last_acc,
                "old_acc": random_old_acc,
                "new_acc": random_new_acc,
                "top1_curve": random_top1_curve,
            }

            rows.append(
                {
                    "alpha": float(alpha),
                    "k": int(k),
                    "effective_k": int(effective_k),
                    "k_capped": bool(k_capped),
                    "method": method_norm,
                    "avg_acc": primary_metrics["avg_acc"],
                    "last_acc": primary_metrics["last_acc"],
                    "old_acc": primary_metrics["old_acc"],
                    "new_acc": primary_metrics["new_acc"],
                    "pca_avg_acc": pca_metrics["avg_acc"] if pca_metrics is not None else None,
                    "pca_last_acc": pca_metrics["last_acc"] if pca_metrics is not None else None,
                    "pca_old_acc": pca_metrics["old_acc"] if pca_metrics is not None else None,
                    "pca_new_acc": pca_metrics["new_acc"] if pca_metrics is not None else None,
                    "explained_variance_ratio": evr_sum,
                    "random_avg_acc": random_avg_acc,
                    "random_avg_acc_std": random_avg_acc_std,
                    "random_last_acc": random_last_acc,
                    "random_last_acc_std": random_last_acc_std,
                    "random_old_acc": random_old_acc,
                    "random_new_acc": random_new_acc,
                    "top1_curve": primary_metrics["top1_curve"],
                }
            )

    summary = _select_summary_records(rows, merged_dim=merged_dim)

    return {
        "rows": rows,
        "alpha_grid": list(alphas),
        "k_grid": list(ks),
        "method": method_norm,
        "random_seeds": list(seeds),
        "pca_backend": pca_backend_used,
        "n_stage0_samples": n_stage0,
        "feature_dim": feat_dim,
        "merged_dim": merged_dim,
        "max_pca_rank": max_pca_rank,
        "increments": increments,
        "metadata": dict(cached_features.get("metadata", {})),
        "summary": summary,
    }


_CSV_FIELDS = [
    "dataset",
    "model_name",
    "method_name",
    "backbone_type",
    "seed",
    "init_cls",
    "increment",
    "alpha",
    "k",
    "effective_k",
    "k_capped",
    "method",
    "avg_acc",
    "last_acc",
    "old_acc",
    "new_acc",
    "pca_avg_acc",
    "pca_last_acc",
    "pca_old_acc",
    "pca_new_acc",
    "explained_variance_ratio",
    "random_avg_acc",
    "random_avg_acc_std",
    "random_last_acc",
    "random_last_acc_std",
    "random_old_acc",
    "random_new_acc",
]


def export_sweep_csv(sweep_result: Dict[str, Any], csv_path: str) -> str:
    """Write 2D sensitivity sweep results to a structured CSV file."""
    abs_path = os.path.abspath(csv_path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    metadata = sweep_result.get("metadata", {})
    seed_val = metadata.get("seed", 1993)
    if isinstance(seed_val, (list, tuple)) and seed_val:
        seed_val = seed_val[0]

    meta_cols = {
        "dataset": metadata.get("dataset", "unknown"),
        "model_name": metadata.get("model_name", "unknown"),
        "method_name": metadata.get("method_name", "unknown"),
        "backbone_type": metadata.get("backbone_type", "unknown"),
        "seed": seed_val,
        "init_cls": metadata.get("init_cls", ""),
        "increment": metadata.get("increment", ""),
    }

    with open(abs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in sweep_result.get("rows", []):
            record = {**meta_cols}
            for field in _CSV_FIELDS:
                if field not in record:
                    record[field] = row.get(field)
            writer.writerow(record)

    return abs_path


def _prepare_plot_data(sweep_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized arrays and quick-read summary highlights for all three Experiment 3 plots."""
    rows = sweep_result.get("rows", [])
    if not rows:
        raise ValueError("sweep_result['rows'] must be non-empty to generate plots.")

    metadata = sweep_result.get("metadata", {})
    dataset = str(metadata.get("dataset", "dataset"))
    method_name = str(metadata.get("method_name", "aper_adapter"))
    alphas = sorted({float(r["alpha"]) for r in rows})
    ks = sorted({int(r["k"]) for r in rows})
    merged_dim = int(sweep_result.get("merged_dim", max(ks)))
    full_k = merged_dim if merged_dim in ks else max(ks)

    row_map = {(float(r["alpha"]), int(r["k"])): r for r in rows}

    # 1. Alpha curve at full dimension
    alpha_rows = sorted([r for r in rows if int(r["k"]) == full_k], key=lambda r: r["alpha"])
    alphas_full = [float(r["alpha"]) for r in alpha_rows]
    avg_full = [float(r["avg_acc"]) for r in alpha_rows]
    last_full = [float(r["last_acc"]) for r in alpha_rows]
    best_full_row = max(alpha_rows, key=lambda r: (r["avg_acc"], r["last_acc"]))

    ptm_full_row = next((r for r in alpha_rows if abs(float(r["alpha"]) - 0.0) <= 1e-6), None)
    default_full_row = next((r for r in alpha_rows if abs(float(r["alpha"]) - 0.5) <= 1e-6), None)
    adapted_full_row = next((r for r in alpha_rows if abs(float(r["alpha"]) - 1.0) <= 1e-6), None)
    delta_best_vs_default_full = (
        round(float(best_full_row["avg_acc"]) - float(default_full_row["avg_acc"]), 2)
        if default_full_row is not None
        else None
    )

    # 2. PCA vs Random compression curve across k at alpha=0.5 and optimal alpha*
    alpha_05 = min(alphas, key=lambda a: abs(a - 0.5))
    summary = sweep_result.get("summary", {})
    best_overall = summary.get("best_overall") or max(rows, key=lambda r: (r["avg_acc"], r["last_acc"]))
    alpha_star = float(best_overall["alpha"])

    comp_rows_05 = [row_map[(alpha_05, k)] for k in ks if (alpha_05, k) in row_map]
    comp_ks = [int(r["k"]) for r in comp_rows_05]
    comp_k_labels = [
        f"{int(r['k'])}({int(r['effective_k'])})" if r.get("k_capped") else str(int(r["k"]))
        for r in comp_rows_05
    ]
    pca_avg_05 = [float(r["pca_avg_acc"] if r["pca_avg_acc"] is not None else r["avg_acc"]) for r in comp_rows_05]
    rand_avg_05 = [r["random_avg_acc"] for r in comp_rows_05]
    rand_std_05 = [r["random_avg_acc_std"] or 0.0 for r in comp_rows_05]

    comp_rows_star = [row_map[(alpha_star, k)] for k in ks if (alpha_star, k) in row_map]
    pca_avg_star = [float(r["pca_avg_acc"] if r["pca_avg_acc"] is not None else r["avg_acc"]) for r in comp_rows_star]

    simplecil_ptm_row = summary.get("simplecil_ptm")
    simplecil_ptm_avg = float(simplecil_ptm_row["avg_acc"]) if simplecil_ptm_row else None
    default_aper_row = summary.get("default_aper")
    default_aper_avg = float(default_aper_row["avg_acc"]) if default_aper_row else None
    delta_best_overall_vs_default = (
        round(float(best_overall["avg_acc"]) - default_aper_avg, 2)
        if default_aper_avg is not None
        else None
    )
    best_pareto_128 = summary.get("best_pareto_128")
    pareto_dim_reduction_pct = (
        round((1.0 - float(best_pareto_128["effective_k"]) / full_k) * 100.0, 1)
        if best_pareto_128 is not None and full_k > 0
        else None
    )

    # 3. 2D Heatmaps across alpha x k
    avg_mat = np.zeros((len(alphas), len(ks)), dtype=np.float32)
    last_mat = np.zeros((len(alphas), len(ks)), dtype=np.float32)
    eff_k_labels = []
    for k in ks:
        sample_row = next((row_map[(a, k)] for a in alphas if (a, k) in row_map), None)
        if sample_row and sample_row.get("k_capped"):
            eff_k_labels.append(f"{k}({sample_row['effective_k']})")
        else:
            eff_k_labels.append(str(k))

    for i, a in enumerate(alphas):
        for j, k in enumerate(ks):
            r = row_map.get((a, k))
            if r is not None:
                avg_mat[i, j] = float(r["avg_acc"])
                last_mat[i, j] = float(r["last_acc"])

    def _cell_index(row: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        if row is None:
            return None
        row_a = float(row["alpha"])
        row_k = int(row["k"])
        if row_k not in ks:
            return None
        alpha_matches = [idx for idx, val in enumerate(alphas) if abs(val - row_a) <= 1e-6]
        if not alpha_matches:
            return None
        return (alpha_matches[0], ks.index(row_k))

    return {
        "dataset": dataset,
        "method_name": method_name,
        "full_k": full_k,
        "max_pca_rank": int(sweep_result.get("max_pca_rank") or 0),
        "n_stage0_samples": int(sweep_result.get("n_stage0_samples") or 0),
        "any_k_capped": bool(any(r.get("k_capped") for r in rows)),
        "alphas": alphas,
        "ks": ks,
        "k_labels": eff_k_labels,
        "comp_k_labels": comp_k_labels,
        "alphas_full": alphas_full,
        "avg_full": avg_full,
        "last_full": last_full,
        "best_full_alpha": float(best_full_row["alpha"]),
        "best_full_avg": float(best_full_row["avg_acc"]),
        "best_full_last": float(best_full_row["last_acc"]),
        "alpha_0_avg": float(ptm_full_row["avg_acc"]) if ptm_full_row is not None else None,
        "alpha_05_avg": float(default_full_row["avg_acc"]) if default_full_row is not None else None,
        "alpha_1_avg": float(adapted_full_row["avg_acc"]) if adapted_full_row is not None else None,
        "delta_best_vs_default_full": delta_best_vs_default_full,
        "alpha_05": alpha_05,
        "alpha_star": alpha_star,
        "comp_ks": comp_ks,
        "pca_avg_05": pca_avg_05,
        "pca_avg_star": pca_avg_star,
        "rand_avg_05": rand_avg_05,
        "rand_std_05": rand_std_05,
        "simplecil_ptm_avg": simplecil_ptm_avg,
        "default_aper_avg": default_aper_avg,
        "delta_best_overall_vs_default": delta_best_overall_vs_default,
        "best_pareto_128": best_pareto_128,
        "pareto_dim_reduction_pct": pareto_dim_reduction_pct,
        "best_cell_idx": _cell_index(best_overall),
        "default_cell_idx": _cell_index(default_aper_row),
        "avg_mat": avg_mat,
        "last_mat": last_mat,
        "best_overall": best_overall,
    }


def _padded_ylim(values: Sequence[Optional[float]], top_frac: float = 0.28, bot_frac: float = 0.18) -> Tuple[float, float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return (0.0, 100.0)
    v_min, v_max = min(clean), max(clean)
    span = max(v_max - v_min, 1.5)
    return (max(0.0, v_min - max(1.8, bot_frac * span)), min(102.0, v_max + max(2.2, top_frac * span)))


def _ha_for_alpha(alpha: float) -> Tuple[str, int]:
    if alpha <= 0.15:
        return "left", 4
    if alpha >= 0.85:
        return "right", -4
    return "center", 0


def _render_plots_matplotlib(plot_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Render all three Experiment 3 figures using headless matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    dataset = plot_data["dataset"]
    method_name = plot_data["method_name"]

    # 1. alpha_curve.png
    alpha_path = os.path.join(output_dir, "alpha_curve.png")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(plot_data["alphas_full"], plot_data["avg_full"], marker="o", linewidth=2.2, label=r"Avg Acc ($\bar{\mathcal{A}}$)")
    ax.plot(plot_data["alphas_full"], plot_data["last_full"], marker="s", linewidth=2.0, linestyle="--", label=r"Last Acc ($\mathcal{A}_B$)")

    best_a = plot_data["best_full_alpha"]
    best_avg = plot_data["best_full_avg"]
    best_last = plot_data["best_full_last"]
    delta_full = plot_data["delta_best_vs_default_full"]

    default_label = (
        f"APER Default (α=0.5): {plot_data['alpha_05_avg']:.2f}%"
        if plot_data["alpha_05_avg"] is not None
        else "APER Default (α=0.5)"
    )
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.4, label=default_label)
    ax.axvline(
        best_a,
        color="crimson",
        linestyle="-.",
        linewidth=1.4,
        label=f"Best α*={best_a:.1f} ({best_avg:.2f}%)",
    )
    ax.plot([best_a], [best_avg], marker="*", markersize=13, color="crimson", linestyle="None", zorder=5)

    best_ha, best_dx = _ha_for_alpha(best_a)
    if abs(best_a - 0.5) <= 1e-6:
        best_note = f"Best = Default α*={best_a:.1f}\nAvg {best_avg:.2f}% (Last {best_last:.2f}%)"
    else:
        delta_str = f"\n(Δ={delta_full:+.2f}% vs Default)" if delta_full is not None else ""
        best_note = f"Best α*={best_a:.1f}: {best_avg:.2f}%{delta_str}"
    ax.annotate(
        best_note,
        xy=(best_a, best_avg),
        xytext=(best_dx, 16),
        textcoords="offset points",
        ha=best_ha,
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="crimson",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "crimson", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": "crimson", "lw": 1.0},
    )

    anchor_specs = [
        (0.0, plot_data["alpha_0_avg"], "PTM (α=0.0)", "teal"),
        (0.5, plot_data["alpha_05_avg"], "Default (α=0.5)", "dimgray"),
        (1.0, plot_data["alpha_1_avg"], "Adapted (α=1.0)", "navy"),
    ]
    for a_val, a_avg, a_name, a_color in anchor_specs:
        if a_avg is None or abs(a_val - best_a) <= 1e-6:
            continue
        ha, dx = _ha_for_alpha(a_val)
        y_off, va = (-20, "top") if abs(a_val - best_a) < 0.25 or abs(a_val - 0.5) <= 1e-6 else (10, "bottom")
        ax.annotate(
            f"{a_name}: {a_avg:.2f}%",
            xy=(a_val, a_avg),
            xytext=(dx, y_off),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.8,
            color=a_color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": a_color, "alpha": 0.85},
            arrowprops={"arrowstyle": "->", "color": a_color, "lw": 0.8},
        )

    ax.set_xticks(plot_data["alphas_full"])
    ax.set_ylim(*_padded_ylim(plot_data["avg_full"] + plot_data["last_full"], top_frac=0.32, bot_frac=0.22))
    ax.set_xlabel("Mixing Coefficient α (0=PTM, 1=Adapted)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{dataset} ({method_name}): α-Mixing Trade-off @ k={plot_data['full_k']}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8.2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(alpha_path, dpi=200)
    plt.close(fig)

    # 2. pca_compression_curve.png
    pca_path = os.path.join(output_dir, "pca_compression_curve.png")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ks = plot_data["comp_ks"]
    ax.plot(ks, plot_data["pca_avg_05"], marker="o", linewidth=2.2, label=f"PCA (α={plot_data['alpha_05']:.1f})")
    has_distinct_star = (
        abs(plot_data["alpha_star"] - plot_data["alpha_05"]) > 1e-6
        and len(plot_data["pca_avg_star"]) == len(ks)
    )
    if has_distinct_star:
        ax.plot(ks, plot_data["pca_avg_star"], marker="^", linewidth=2.0, label=f"PCA (α*={plot_data['alpha_star']:.1f})")

    rand_bounds: List[float] = []
    if all(v is not None for v in plot_data["rand_avg_05"]):
        r_mean = np.asarray(plot_data["rand_avg_05"], dtype=np.float32)
        r_std = np.asarray(plot_data["rand_std_05"], dtype=np.float32)
        ax.plot(ks, r_mean, marker="x", linestyle="--", linewidth=1.8, color="darkorange", label=f"Random Subsampling (α={plot_data['alpha_05']:.1f})")
        ax.fill_between(ks, r_mean - r_std, r_mean + r_std, color="darkorange", alpha=0.2)
        rand_bounds.extend((r_mean - r_std).tolist())
        rand_bounds.extend((r_mean + r_std).tolist())

    if plot_data["simplecil_ptm_avg"] is not None:
        ax.axhline(plot_data["simplecil_ptm_avg"], color="teal", linestyle=":", linewidth=1.5, label=f"SimpleCIL-PTM ({plot_data['simplecil_ptm_avg']:.2f}%)")
    if plot_data["default_aper_avg"] is not None:
        ax.axhline(plot_data["default_aper_avg"], color="gray", linestyle="-.", linewidth=1.2, label=f"Full APER ({plot_data['default_aper_avg']:.2f}%)")

    if plot_data["any_k_capped"] and plot_data["max_pca_rank"] > 0:
        ax.axvline(
            plot_data["max_pca_rank"],
            color="purple",
            linestyle=":",
            linewidth=1.6,
            label=f"Stage-1 PCA rank cap (r={plot_data['max_pca_rank']}, |D1|={plot_data['n_stage0_samples']})",
        )

    pareto = plot_data["best_pareto_128"]
    if pareto is not None and int(pareto["k"]) < plot_data["full_k"]:
        p_k = int(pareto["k"])
        p_avg = float(pareto["avg_acc"])
        dim_red = plot_data["pareto_dim_reduction_pct"]
        red_str = f"-{dim_red:.0f}% dim" if dim_red is not None else f"eff={pareto['effective_k']}"
        delta_str = (
            f" ({p_avg - plot_data['default_aper_avg']:+.2f}% vs Full)"
            if plot_data["default_aper_avg"] is not None
            else ""
        )
        ax.plot([p_k], [p_avg], marker="*", markersize=12, color="darkgreen", linestyle="None", zorder=5)
        ax.annotate(
            f"Pareto α={pareto['alpha']:.1f}, k={p_k} ({red_str}): {p_avg:.2f}%{delta_str}",
            xy=(p_k, p_avg),
            xytext=(6, 14),
            textcoords="offset points",
            ha="left" if p_k <= ks[max(len(ks) // 2 - 1, 0)] else "right",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="darkgreen",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "darkgreen", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "darkgreen", "lw": 0.9},
        )

    if len(ks) > 1 and min(ks) > 0:
        ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(plot_data["comp_k_labels"], rotation=25, ha="right", fontsize=8.5)
    y_candidates: List[Optional[float]] = (
        list(plot_data["pca_avg_05"])
        + (list(plot_data["pca_avg_star"]) if has_distinct_star else [])
        + rand_bounds
        + [plot_data["simplecil_ptm_avg"], plot_data["default_aper_avg"]]
        + ([float(pareto["avg_acc"])] if pareto is not None else [])
    )
    ax.set_ylim(*_padded_ylim(y_candidates, top_frac=0.30, bot_frac=0.18))
    ax.set_xlabel(
        "Projected Dimension k (parentheses = Stage-1 capped rank)"
        if plot_data["any_k_capped"]
        else "Projected Feature Dimension k"
    )
    ax.set_ylabel("Average Incremental Accuracy (%)")
    ax.set_title(f"{dataset} ({method_name}): PCA Projection vs Random Subsampling")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8.0, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(pca_path, dpi=200)
    plt.close(fig)

    # 3. heatmap_alpha_k.png
    heatmap_path = os.path.join(output_dir, "heatmap_alpha_k.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), sharey=True)
    panels = [
        (axes[0], plot_data["avg_mat"], r"Avg Acc ($\bar{\mathcal{A}}$ %)"),
        (axes[1], plot_data["last_mat"], r"Last Acc ($\mathcal{A}_B$ %)"),
    ]
    best_cell = plot_data["best_cell_idx"]
    default_cell = plot_data["default_cell_idx"]

    for ax_panel, mat, title in panels:
        im = ax_panel.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
        ax_panel.set_xticks(range(len(plot_data["ks"])))
        ax_panel.set_xticklabels(plot_data["k_labels"], rotation=30, ha="right", fontsize=8)
        ax_panel.set_yticks(range(len(plot_data["alphas"])))
        ax_panel.set_yticklabels([f"{a:.1f}" for a in plot_data["alphas"]], fontsize=8.5)
        ax_panel.set_xlabel("Dimension k (effective if capped)")
        ax_panel.set_title(title)
        vmax = float(np.max(mat)) if mat.size else 1.0
        vmin = float(np.min(mat)) if mat.size else 0.0
        mid = 0.5 * (vmax + vmin)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                is_best = best_cell is not None and (i, j) == best_cell
                color = "black" if val >= mid else "white"
                ax_panel.text(
                    j,
                    i,
                    f"{val:.1f}*" if is_best else f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                    fontweight="bold" if is_best else "normal",
                )
        if best_cell is not None:
            i_b, j_b = best_cell
            ax_panel.add_patch(
                Rectangle((j_b - 0.5, i_b - 0.5), 1.0, 1.0, fill=False, edgecolor="#ff2a2a", linewidth=2.5, zorder=4)
            )
        if default_cell is not None and default_cell != best_cell:
            i_d, j_d = default_cell
            ax_panel.add_patch(
                Rectangle((j_d - 0.5, i_d - 0.5), 1.0, 1.0, fill=False, edgecolor="cyan", linewidth=2.0, linestyle="--", zorder=3)
            )
        fig.colorbar(im, ax=ax_panel, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Mixing Coefficient α")
    best = plot_data["best_overall"]
    delta_overall = plot_data["delta_best_overall_vs_default"]
    default_avg = plot_data["default_aper_avg"]
    line1 = (
        f"{dataset} ({method_name}) 2D Sensitivity — Best (α*={best['alpha']:.1f}, "
        f"k*={best['k']}, eff={best['effective_k']}) => Avg {best['avg_acc']:.2f}%, Last {best['last_acc']:.2f}%"
    )
    if default_cell is not None and default_cell == best_cell:
        box_legend = "[Red box = Best* = Default APER]"
    else:
        box_legend = "[Red box = Best*, Cyan dashed = Default APER]"
    if default_avg is not None and delta_overall is not None:
        line2 = (
            f"Default APER (α=0.5, k={plot_data['full_k']}): Avg {default_avg:.2f}% | "
            f"ΔAvg={delta_overall:+.2f}%   {box_legend}"
        )
    else:
        line2 = f"Default APER reference highlighted when present   {box_legend}"
    fig.suptitle(f"{line1}\n{line2}", fontsize=10.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    return {
        "alpha_curve": alpha_path,
        "pca_compression_curve": pca_path,
        "heatmap_alpha_k": heatmap_path,
    }


def _value_to_viridis_rgb(t: float) -> Tuple[int, int, int]:
    """Map normalized [0, 1] scalar to an RGB color gradient."""
    t_c = max(0.0, min(1.0, float(t)))
    r = int(68 + t_c * (253 - 68))
    g = int(1 + math.sin(t_c * math.pi * 0.65) * 185 + t_c * 45)
    b = int(84 + (1.0 - t_c) * 90)
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


def _render_plots_pillow(plot_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Headless Pillow fallback renderer when matplotlib is not installed locally."""
    from PIL import Image, ImageDraw

    def _draw_line_chart(path: str, title: str, x_vals: List[float], series: List[Tuple[str, List[float], Tuple[int, int, int]]], x_labels: Optional[List[str]] = None):
        w, h = 840, 520
        pad_l, pad_r, pad_t, pad_b = 75, 35, 55, 70
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((pad_l, 18), title, fill=(20, 20, 20))
        draw.rectangle([pad_l, pad_t, w - pad_r, h - pad_b], outline=(80, 80, 80), width=2)

        all_y = [v for _, ys, _ in series for v in ys if v is not None]
        y_min = min(all_y) - 1.0 if all_y else 0.0
        y_max = max(all_y) + 1.0 if all_y else 100.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        n_pts = max(len(x_vals), 1)
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        def _xy(idx: int, y_val: float) -> Tuple[int, int]:
            px = pad_l + int((idx / max(n_pts - 1, 1)) * plot_w) if n_pts > 1 else pad_l + plot_w // 2
            py = h - pad_b - int(((y_val - y_min) / (y_max - y_min)) * plot_h)
            return px, py

        for idx, x_val in enumerate(x_vals):
            lbl = x_labels[idx] if x_labels else f"{x_val:.1f}"
            px, _ = _xy(idx, y_min)
            draw.line([(px, pad_t), (px, h - pad_b)], fill=(230, 230, 230), width=1)
            draw.text((px - 10, h - pad_b + 10), lbl, fill=(40, 40, 40))

        for s_idx, (name, ys, color) in enumerate(series):
            pts = [_xy(i, y) for i, y in enumerate(ys) if y is not None]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=3)
            for px, py in pts:
                draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
            draw.text((pad_l + 15 + s_idx * 190, h - 28), name, fill=color)

        img.save(path, format="PNG")

    dataset = plot_data["dataset"]
    method_name = plot_data["method_name"]

    alpha_path = os.path.join(output_dir, "alpha_curve.png")
    _draw_line_chart(
        alpha_path,
        f"{dataset} ({method_name}) alpha-Mixing @ k={plot_data['full_k']}",
        plot_data["alphas_full"],
        [
            ("Avg Acc", plot_data["avg_full"], (31, 119, 180)),
            ("Last Acc", plot_data["last_full"], (214, 39, 40)),
        ],
    )

    pca_path = os.path.join(output_dir, "pca_compression_curve.png")
    comp_series = [("PCA (a=0.5)", plot_data["pca_avg_05"], (31, 119, 180))]
    if all(v is not None for v in plot_data["rand_avg_05"]):
        comp_series.append(("Random Subsampling", [float(v) for v in plot_data["rand_avg_05"]], (255, 127, 14)))
    _draw_line_chart(
        pca_path,
        f"{dataset} ({method_name}) PCA vs Random Compression",
        list(range(len(plot_data["comp_ks"]))),
        comp_series,
        x_labels=plot_data["comp_k_labels"],
    )

    heatmap_path = os.path.join(output_dir, "heatmap_alpha_k.png")
    avg_mat = plot_data["avg_mat"]
    n_rows, n_cols = avg_mat.shape
    w, h = 920, 560
    pad_l, pad_r, pad_t, pad_b = 80, 40, 60, 60
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    best = plot_data["best_overall"]
    delta_overall = plot_data.get("delta_best_overall_vs_default")
    delta_suffix = f" (dAvg={delta_overall:+.2f}%)" if delta_overall is not None else ""
    draw.text(
        (pad_l, 20),
        f"{dataset} ({method_name}) 2D Heatmap Avg Acc - Best (a={best['alpha']:.1f}, k={best['k']}): {best['avg_acc']:.2f}%{delta_suffix}",
        fill=(20, 20, 20),
    )

    cell_w = max((w - pad_l - pad_r) // max(n_cols, 1), 1)
    cell_h = max((h - pad_t - pad_b) // max(n_rows, 1), 1)
    vmin = float(np.min(avg_mat)) if avg_mat.size else 0.0
    vmax = float(np.max(avg_mat)) if avg_mat.size else 1.0
    span = max(vmax - vmin, 1e-6)
    best_cell = plot_data.get("best_cell_idx")

    for i in range(n_rows):
        y0 = pad_t + (n_rows - 1 - i) * cell_h
        draw.text((25, y0 + cell_h // 3), f"{plot_data['alphas'][i]:.1f}", fill=(30, 30, 30))
        for j in range(n_cols):
            x0 = pad_l + j * cell_w
            val = float(avg_mat[i, j])
            rgb = _value_to_viridis_rgb((val - vmin) / span)
            is_best = best_cell is not None and (i, j) == best_cell
            outline_col = (220, 30, 30) if is_best else (255, 255, 255)
            draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + cell_h - 2], fill=rgb, outline=outline_col, width=3 if is_best else 1)
            draw.text((x0 + 6, y0 + cell_h // 3), f"{val:.1f}*" if is_best else f"{val:.1f}", fill=(255, 255, 255))
            if i == 0:
                draw.text((x0 + 8, h - pad_b + 10), plot_data["k_labels"][j], fill=(30, 30, 30))

    img.save(heatmap_path, format="PNG")
    return {
        "alpha_curve": alpha_path,
        "pca_compression_curve": pca_path,
        "heatmap_alpha_k": heatmap_path,
    }


def generate_sensitivity_plots(sweep_result: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Generate alpha_curve.png, pca_compression_curve.png, and heatmap_alpha_k.png.
    Uses headless matplotlib when installed and falls back to Pillow otherwise.
    """
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    plot_data = _prepare_plot_data(sweep_result)

    try:
        import matplotlib  # noqa: F401
        return _render_plots_matplotlib(plot_data, abs_dir)
    except ImportError:
        logging.warning("matplotlib is not installed; falling back to headless Pillow chart renderer.")
        return _render_plots_pillow(plot_data, abs_dir)


def log_sensitivity_to_wandb(
    sweep_result: Dict[str, Any],
    args: Dict[str, Any],
    plot_paths: Optional[Dict[str, str]] = None,
    csv_path: Optional[str] = None,
):
    """
    Optionally log the 2D sensitivity sweep table, optimal (alpha*, k*) summary metrics,
    and generated figures to Weights & Biases (`nnm-hcmus/Group7-ML-Lab`).
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

    metadata = sweep_result.get("metadata", {})
    merged_args = {**metadata, **args}
    backbone_type = args.get("backbone_type") or args.get("convnet_type") or metadata.get("backbone_type")
    if backbone_type and not merged_args.get("backbone_type"):
        merged_args["backbone_type"] = backbone_type

    method_name = metadata.get("method_name") or _resolve_method_name(merged_args)
    dataset = args.get("dataset") or metadata.get("dataset", "unknown")
    init_cls = args.get("init_cls") if args.get("init_cls") is not None else metadata.get("init_cls", 0)
    increment = args.get("increment") if args.get("increment") is not None else metadata.get("increment", 10)
    seed_raw = args.get("seed") if args.get("seed") is not None else metadata.get("seed", 1993)
    seed = seed_raw[0] if isinstance(seed_raw, (list, tuple)) and seed_raw else seed_raw

    tags = _resolve_wandb_tags(merged_args)
    entity = args.get("wandb_entity") or "nnm-hcmus"
    project = args.get("wandb_project") or "Group7-ML-Lab"
    group = args.get("wandb_group") or f"exp3_{dataset}_init{init_cls}_inc{increment}"
    name = args.get("wandb_name") or f"exp3_{method_name}_{dataset}_B{init_cls}_Inc{increment}_seed{seed}"

    total_params = int(metadata.get("total_params", 0))
    trainable_params = int(metadata.get("trainable_params", 0))

    config: Dict[str, Any] = {}
    for key, value in merged_args.items():
        if key in ("convnet_type", "wandb_tags"):
            continue
        if key == "device" and isinstance(value, (list, tuple)):
            config[key] = [str(d) for d in value]
        else:
            config[key] = value
    if backbone_type:
        config["backbone_type"] = backbone_type
    config["method_name"] = method_name
    config["total_params"] = total_params
    config["trainable_params"] = trainable_params
    config["alpha_grid"] = sweep_result.get("alpha_grid", [])
    config["k_grid"] = sweep_result.get("k_grid", [])
    config["n_stage0_samples"] = sweep_result.get("n_stage0_samples", 0)
    config["pca_backend"] = sweep_result.get("pca_backend")

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

        columns = [
            "alpha",
            "k",
            "effective_k",
            "k_capped",
            "method",
            "avg_acc",
            "last_acc",
            "old_acc",
            "new_acc",
            "pca_avg_acc",
            "pca_last_acc",
            "explained_variance_ratio",
            "random_avg_acc",
            "random_avg_acc_std",
            "random_last_acc",
            "random_last_acc_std",
        ]
        table_data = [[row.get(col) for col in columns] for row in sweep_result.get("rows", [])]
        table = wandb.Table(columns=columns, data=table_data)

        log_payload: Dict[str, Any] = {"exp3/sensitivity_table": table}
        if plot_paths:
            for key in ("alpha_curve", "pca_compression_curve", "heatmap_alpha_k"):
                path = plot_paths.get(key)
                if path and os.path.exists(path):
                    log_payload[f"exp3/{key}"] = wandb.Image(path)

        if hasattr(wandb, "log"):
            wandb.log(log_payload)
        elif run is not None and hasattr(run, "log"):
            run.log(log_payload)

        summary_target = getattr(wandb, "run", None) or run
        if summary_target is not None and getattr(summary_target, "summary", None) is not None:
            summary = sweep_result.get("summary", {})
            best = summary.get("best_overall")
            best_full = summary.get("best_full_dim")
            default_aper = summary.get("default_aper")
            if best:
                summary_target.summary["best_alpha"] = best["alpha"]
                summary_target.summary["best_k"] = best["k"]
                summary_target.summary["best_effective_k"] = best["effective_k"]
                summary_target.summary["best_avg_acc"] = best["avg_acc"]
                summary_target.summary["best_last_acc"] = best["last_acc"]
                summary_target.summary["avg_acc"] = best["avg_acc"]
                summary_target.summary["last_acc"] = best["last_acc"]
            if best_full:
                summary_target.summary["full_dim_best_alpha"] = best_full["alpha"]
                summary_target.summary["full_dim_best_avg_acc"] = best_full["avg_acc"]
                summary_target.summary["full_dim_best_last_acc"] = best_full["last_acc"]
            if default_aper:
                summary_target.summary["default_aper_avg_acc"] = default_aper["avg_acc"]
                summary_target.summary["default_aper_last_acc"] = default_aper["last_acc"]
            summary_target.summary["total_params"] = total_params
            summary_target.summary["trainable_params"] = trainable_params
            if csv_path:
                summary_target.summary["csv_path"] = csv_path
        return run
    finally:
        if run is not None:
            if hasattr(wandb, "finish"):
                wandb.finish()
            elif hasattr(run, "finish"):
                run.finish()
