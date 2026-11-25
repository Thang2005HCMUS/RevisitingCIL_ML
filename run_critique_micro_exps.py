import argparse
import copy
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from analysis.critique_audits import (
    benchmark_inference_variants,
    export_critique_csvs,
    export_critique_json,
    generate_critique_plots,
    log_critique_to_wandb,
    run_norm_merge_audit,
)


def load_json(settings_path: str) -> Dict[str, Any]:
    """Read experiment settings from a JSON config file."""
    with open(settings_path, "r", encoding="utf-8") as data_file:
        return json.load(data_file)


def setup_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the critique empirical micro-experiments."""
    parser = argparse.ArgumentParser(
        description=(
            "Critique Empirical Micro-Experiments: "
            "(1) L2 Norm Imbalance & Unnormalized vs. Normalized Merge Audit, "
            "(2) 1-Branch vs. 2-Branch Inference Latency & Memory Benchmark."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["norm_audit", "latency_bench", "all"],
        default="all",
        help="Micro-experiment mode to run (default: all).",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to an Exp 3 dual-branch feature cache (.pt).",
    )
    parser.add_argument(
        "--cache_paths",
        nargs="+",
        default=None,
        help="Optional list of Exp 3 feature cache (.pt) paths to audit in one run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./exps/aper_adapter/cifar224_5_5_vit-b_adapter.json",
        help="Config JSON used to instantiate real pretrained backbones for latency_bench.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/exp4_critique_audits",
        help="Directory to write CSV tables, JSON summary, and PNG figures.",
    )
    parser.add_argument(
        "--best_alpha",
        type=float,
        default=1.0,
        help="Reference optimal alpha in [0, 1] for normalized_best_alpha comparison (default: 1.0).",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=128,
        help="Linear projection dimension k for APER + PCA inference benchmark (default: 128).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference latency and memory benchmark (default: 64).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=200,
        help="Number of prototype classes in CosinePrototypeClassifier (default: 200).",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of untimed warmup iterations before latency timing (default: 10).",
    )
    parser.add_argument(
        "--timed_iters",
        type=int,
        default=50,
        help="Number of timed forward iterations for latency measurement (default: 50).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device ('auto', 'cpu', 'cuda', or 'cuda:0').",
    )
    parser.add_argument(
        "--backbone_source",
        type=str,
        choices=["pretrained", "mock"],
        default="pretrained",
        help="Backbone source for latency_bench: 'pretrained' ViT or lightweight 'mock' modules.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging with ['empirics', 'critique-audit'] tags.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="nnm-hcmus",
        help="Weights & Biases entity (default: nnm-hcmus).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Group7-ML-Lab",
        help="Weights & Biases project (default: Group7-ML-Lab).",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Weights & Biases run group.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb_tags",
        nargs="+",
        default=None,
        help="Additional Weights & Biases tags appended after ['empirics', 'critique-audit'].",
    )
    return parser


class _MockViTBackbone(nn.Module):
    """Lightweight deterministic mock backbone exposing out_dim=768 for local CPU verification."""

    def __init__(self, out_dim: int = 768, seed: int = 1993):
        super().__init__()
        self.out_dim = int(out_dim)
        self.stem = nn.Conv2d(3, 16, kernel_size=16, stride=16, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(16, self.out_dim, bias=False)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.stem.weight.copy_(torch.randn(self.stem.weight.shape, generator=gen) * 0.05)
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=gen) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.stem(x)).flatten(1)
        return self.proj(h)


def build_mock_backbones(out_dim: int = 768) -> Tuple[nn.Module, nn.Module]:
    """Construct lightweight frozen and adapted mock backbones without downloading weights."""
    return _MockViTBackbone(out_dim=out_dim, seed=1993), _MockViTBackbone(out_dim=out_dim, seed=2026)


def _strip_pet_suffix(convnet_type: str) -> str:
    """Resolve the plain pretrained PTM name corresponding to a PET backbone string."""
    base = str(convnet_type or "pretrained_vit_b16_224_in21k")
    for suffix in ("_adapter", "_ssf", "_vpt", "_bn"):
        base = base.replace(suffix, "")
    return base


def build_pretrained_backbones(config_path: Optional[str]) -> Tuple[nn.Module, nn.Module, str]:
    """Instantiate real frozen PTM and adapted PTM backbones from a JSON config."""
    from models.backbones import get_pretrained_model

    cfg: Dict[str, Any] = {}
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Backbone config file not found at '{config_path}'.")
        cfg = load_json(config_path)

    adapted_type = str(cfg.get("convnet_type") or cfg.get("backbone_type") or "pretrained_vit_b16_224_in21k_adapter")
    frozen_cfg = copy.deepcopy(cfg)
    frozen_cfg["convnet_type"] = _strip_pet_suffix(adapted_type)
    adapted_cfg = copy.deepcopy(cfg)
    adapted_cfg["convnet_type"] = adapted_type

    try:
        frozen_ptm = get_pretrained_model(frozen_cfg)
        adapted_ptm = get_pretrained_model(adapted_cfg)
    except (ImportError, ModuleNotFoundError) as exc:
        if "timm.layers" in str(exc):
            raise RuntimeError(
                f"Failed to instantiate pretrained PET backbone '{adapted_type}' because the installed `timm` "
                "version lacks `timm.layers` (Colab requires `timm >= 1.0`). Use `--backbone_source mock` for "
                "local CPU smoke checks."
            ) from exc
        raise
    return frozen_ptm, adapted_ptm, adapted_type


def _resolve_cache_paths(args_dict: Dict[str, Any]) -> List[str]:
    """Collect and validate unique feature cache paths from CLI arguments."""
    paths: List[str] = []
    if args_dict.get("cache_path"):
        paths.append(str(args_dict["cache_path"]))
    for p in args_dict.get("cache_paths") or []:
        p_str = str(p)
        if p_str not in paths:
            paths.append(p_str)
    return paths


def run_critique_micro_experiments(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute requested critique micro-experiments, save artifacts, and optionally log to W&B."""
    run_args = copy.deepcopy(args_dict)
    mode = str(run_args.get("mode") or "all").lower()
    if mode not in ("norm_audit", "latency_bench", "all"):
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of ('norm_audit', 'latency_bench', 'all').")

    output_dir = str(run_args.get("output_dir") or os.path.join("experiments", "exp4_critique_audits"))
    cache_paths = _resolve_cache_paths(run_args)

    if mode in ("norm_audit", "all"):
        if not cache_paths:
            raise ValueError(
                f"--mode {mode} requires --cache_path or --cache_paths pointing to an Exp 3 feature cache "
                "(e.g. experiments/exp3_analysis/cifar224/cifar224_seed1993_features.pt)."
            )
        missing_caches = [p for p in cache_paths if not os.path.exists(p)]
        if missing_caches:
            raise FileNotFoundError(
                f"Missing feature cache file(s): {missing_caches}. "
                "Generate them via `run_exp3_sensitivity.py` or pass valid paths."
            )

    norm_audits: List[Dict[str, Any]] = []
    if mode in ("norm_audit", "all"):
        best_alpha = float(run_args.get("best_alpha", 1.0))
        for path in cache_paths:
            audit = run_norm_merge_audit(path, best_alpha=best_alpha)
            norm_audits.append(audit)
            summary = audit["summary"]
            logging.info(
                f"[Norm Audit | {audit['dataset']}] mean norm ratio={summary['test_mean_norm_ratio']:.4f}, "
                f"alpha_eff={summary['test_mean_alpha_eff']:.4f}, "
                f"unnorm_avg={summary['unnormalized_avg_acc']:.2f}%, "
                f"norm(0.5)_avg={summary['normalized_alpha_0_5_avg_acc']:.2f}% "
                f"(delta={summary['delta_alpha_0_5_vs_unnorm']:+.2f}%)"
            )

    latency_result: Optional[Dict[str, Any]] = None
    if mode in ("latency_bench", "all"):
        backbone_source = str(run_args.get("backbone_source") or "pretrained").lower()
        if backbone_source == "mock":
            frozen_ptm, adapted_ptm = build_mock_backbones(out_dim=768)
            run_args.setdefault("backbone_type", "mock_vit_768")
        elif backbone_source == "pretrained":
            frozen_ptm, adapted_ptm, backbone_type = build_pretrained_backbones(run_args.get("config"))
            run_args.setdefault("backbone_type", backbone_type)
        else:
            raise ValueError(f"Unsupported backbone_source '{backbone_source}'. Expected 'pretrained' or 'mock'.")

        latency_result = benchmark_inference_variants(
            frozen_ptm=frozen_ptm,
            adapted_ptm=adapted_ptm,
            batch_size=int(run_args.get("batch_size", 64)),
            num_classes=int(run_args.get("num_classes", 200)),
            pca_dim=int(run_args.get("pca_dim", 128)),
            warmup_iters=int(run_args.get("warmup_iters", 10)),
            timed_iters=int(run_args.get("timed_iters", 50)),
            device=run_args.get("device", "auto"),
        )
        for row in latency_result["variants"]:
            logging.info(
                f"[Latency Bench | {row['label']}] {row['ms_per_batch_mean']:.2f} ms/batch "
                f"({row['images_per_sec']:.1f} img/s), params={row['total_params']:,}, "
                f"peak_mem={row['peak_memory_mb']:.2f} MB ({row['memory_backend']}), "
                f"latency_x={row['latency_ratio_vs_simplecil']:.2f}x"
            )

    os.makedirs(output_dir, exist_ok=True)
    csv_paths = export_critique_csvs(output_dir, norm_audits=norm_audits or None, latency_result=latency_result)
    json_path = export_critique_json(output_dir, norm_audits=norm_audits or None, latency_result=latency_result)
    plot_paths = generate_critique_plots(output_dir, norm_audits=norm_audits or None, latency_result=latency_result)

    run_args["use_wandb"] = bool(run_args.get("wandb") or run_args.get("use_wandb"))
    log_critique_to_wandb(
        run_args,
        norm_audits=norm_audits or None,
        latency_result=latency_result,
        plot_paths=plot_paths,
        csv_paths=csv_paths,
        json_path=json_path,
    )

    return {
        "mode": mode,
        "output_dir": os.path.abspath(output_dir),
        "norm_audits": norm_audits,
        "latency_result": latency_result,
        "csv_paths": csv_paths,
        "json_path": json_path,
        "plot_paths": plot_paths,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s")
    args = setup_parser().parse_args()
    return run_critique_micro_experiments(vars(args))


if __name__ == "__main__":
    main()
