import csv
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analysis.critique_audits import (
    DEFAULT_CRITIQUE_WANDB_TAGS,
    benchmark_inference_variants,
    compute_norm_imbalance_stats,
    export_critique_csvs,
    export_critique_json,
    generate_critique_plots,
    log_critique_to_wandb,
    run_norm_merge_audit,
)
from analysis.sensitivity_sweep import save_feature_cache
import run_critique_micro_exps


class _TinyConvBackbone(nn.Module):
    def __init__(self, out_dim: int = 32, seed: int = 0):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, out_dim, bias=False)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=gen))
            self.fc.weight.copy_(torch.randn(self.fc.weight.shape, generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(x)).flatten(1))


def _make_synthetic_cache(
    adapted_scale: float = 1.0,
    frozen_scale: float = 1.0,
    dim: int = 8,
    n_classes: int = 4,
    samples_per_class: int = 6,
) -> dict:
    rng = np.random.RandomState(1993)
    class_centers_ad = rng.randn(n_classes, dim).astype(np.float32)
    class_centers_fr = rng.randn(n_classes, dim).astype(np.float32)

    # Make adapted and frozen branches prefer complementary discriminability so alpha shifts matter.
    class_centers_ad[0] = class_centers_ad[1]  # adapted confuses class 0 and 1
    class_centers_fr[2] = class_centers_fr[3]  # frozen confuses class 2 and 3

    tr_ad, tr_fr, tr_y = [], [], []
    te_ad, te_fr, te_y = [], [], []
    for cls_id in range(n_classes):
        for _ in range(samples_per_class):
            vec_ad = class_centers_ad[cls_id] + 0.05 * rng.randn(dim).astype(np.float32)
            vec_fr = class_centers_fr[cls_id] + 0.05 * rng.randn(dim).astype(np.float32)
            vec_ad = vec_ad / np.linalg.norm(vec_ad) * adapted_scale
            vec_fr = vec_fr / np.linalg.norm(vec_fr) * frozen_scale
            tr_ad.append(vec_ad)
            tr_fr.append(vec_fr)
            tr_y.append(cls_id)

            t_ad = class_centers_ad[cls_id] + 0.05 * rng.randn(dim).astype(np.float32)
            t_fr = class_centers_fr[cls_id] + 0.05 * rng.randn(dim).astype(np.float32)
            t_ad = t_ad / np.linalg.norm(t_ad) * adapted_scale
            t_fr = t_fr / np.linalg.norm(t_fr) * frozen_scale
            te_ad.append(t_ad)
            te_fr.append(t_fr)
            te_y.append(cls_id)

    return {
        "train_adapted": np.stack(tr_ad, axis=0),
        "train_frozen": np.stack(tr_fr, axis=0),
        "train_targets": np.asarray(tr_y, dtype=np.int64),
        "test_adapted": np.stack(te_ad, axis=0),
        "test_frozen": np.stack(te_fr, axis=0),
        "test_targets": np.asarray(te_y, dtype=np.int64),
        "increments": [2, 2],
        "task_class_ids": [[0, 1], [2, 3]],
        "stage0_classes": [0, 1],
        "metadata": {
            "dataset": "synthetic_cifar",
            "model_name": "aper_adapter",
            "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
            "seed": 1993,
            "init_cls": 2,
            "increment": 2,
        },
    }


class TestCritiqueMicroExps(unittest.TestCase):
    def test_compute_norm_imbalance_stats_analytical_and_scaling(self):
        cache = {
            "train_adapted": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
            "train_frozen": np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32),
            "train_targets": np.array([0, 1], dtype=np.int64),
            "test_adapted": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
            "test_frozen": np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32),
            "test_targets": np.array([0, 1], dtype=np.int64),
            "metadata": {"dataset": "analytical"},
        }
        stats = compute_norm_imbalance_stats(cache)
        self.assertAlmostEqual(stats["train"]["adapted_norm"]["mean"], 3.0, places=6)
        self.assertAlmostEqual(stats["train"]["frozen_norm"]["mean"], 4.0, places=6)
        self.assertAlmostEqual(stats["train"]["norm_ratio"]["mean"], 0.75, places=6)
        self.assertAlmostEqual(stats["train"]["alpha_eff"]["mean"], 9.0 / 25.0, places=6)
        self.assertAlmostEqual(stats["test"]["alpha_eff"]["median"], 0.36, places=6)
        self.assertAlmostEqual(stats["prototypes"]["norm_ratio"]["mean"], 0.75, places=6)
        self.assertAlmostEqual(stats["prototypes"]["alpha_eff"]["mean"], 0.36, places=6)

        # Scaling adapted branch by 2.0 -> ||phi*||=6, ||phi||=4 -> r=1.5, alpha_eff=36/52
        scaled_cache = dict(cache)
        scaled_cache["train_adapted"] = cache["train_adapted"] * 2.0
        scaled_cache["test_adapted"] = cache["test_adapted"] * 2.0
        scaled_stats = compute_norm_imbalance_stats(scaled_cache)
        self.assertAlmostEqual(scaled_stats["train"]["norm_ratio"]["mean"], 1.5, places=6)
        self.assertAlmostEqual(scaled_stats["train"]["alpha_eff"]["mean"], 36.0 / 52.0, places=6)

    def test_compute_norm_imbalance_stats_zero_norm_and_malformed_cache(self):
        zero_cache = {
            "train_adapted": np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "train_frozen": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            "train_targets": np.array([0, 1], dtype=np.int64),
            "test_adapted": np.array([[0.0, 0.0]], dtype=np.float32),
            "test_frozen": np.array([[5.0, 0.0]], dtype=np.float32),
            "test_targets": np.array([0], dtype=np.int64),
        }
        stats = compute_norm_imbalance_stats(zero_cache, eps=1e-12)
        self.assertEqual(stats["train"]["zero_adapted_count"], 1)
        self.assertEqual(stats["train"]["zero_frozen_count"], 2)
        self.assertEqual(stats["test"]["zero_adapted_count"], 1)
        for split in ("train", "test", "prototypes"):
            for metric_key in ("adapted_norm", "frozen_norm", "norm_ratio", "alpha_eff"):
                for stat_val in stats[split][metric_key].values():
                    self.assertTrue(np.isfinite(stat_val))
        self.assertGreaterEqual(stats["train"]["alpha_eff"]["p05"], 0.0)
        self.assertLessEqual(stats["train"]["alpha_eff"]["p95"], 1.0)

        with self.assertRaises(ValueError):
            compute_norm_imbalance_stats({"train_adapted": np.ones((2, 2))})

    def test_run_norm_merge_audit_scale_invariance_and_deltas(self):
        with self.assertRaises(FileNotFoundError):
            run_norm_merge_audit("nonexistent_dir/missing_cache.pt")

        base_cache = _make_synthetic_cache(adapted_scale=1.0, frozen_scale=1.0)
        dominated_adapted_cache = _make_synthetic_cache(adapted_scale=1e5, frozen_scale=1.0)
        dominated_frozen_cache = _make_synthetic_cache(adapted_scale=1e-5, frozen_scale=1.0)

        res_base = run_norm_merge_audit(base_cache, best_alpha=0.7)
        res_ad = run_norm_merge_audit(dominated_adapted_cache, best_alpha=0.7)
        res_fr = run_norm_merge_audit(dominated_frozen_cache, best_alpha=0.7)

        expected_variants = [
            "unnormalized_concat",
            "normalized_alpha_0.5",
            "normalized_best_alpha",
            "ptm_only",
            "adapted_only",
        ]
        self.assertEqual([v["variant"] for v in res_base["variants"]], expected_variants)

        # Normalized merge accuracy must be invariant to branch norm scaling.
        self.assertEqual(
            res_base["variants_by_name"]["normalized_alpha_0.5"]["avg_acc"],
            res_ad["variants_by_name"]["normalized_alpha_0.5"]["avg_acc"],
        )
        self.assertEqual(
            res_base["variants_by_name"]["normalized_best_alpha"]["avg_acc"],
            res_fr["variants_by_name"]["normalized_best_alpha"]["avg_acc"],
        )

        # Unnormalized concat shifts to adapted_only when adapted norm >> frozen norm,
        # and shifts to ptm_only when adapted norm << frozen norm.
        self.assertGreater(res_ad["norm_stats"]["test_mean_alpha_eff"], 0.9999)
        self.assertLess(res_fr["norm_stats"]["test_mean_alpha_eff"], 0.0001)
        self.assertNotEqual(
            res_ad["variants_by_name"]["unnormalized_concat"]["avg_acc"],
            res_fr["variants_by_name"]["unnormalized_concat"]["avg_acc"],
        )
        self.assertEqual(
            res_ad["variants_by_name"]["unnormalized_concat"]["avg_acc"],
            res_ad["variants_by_name"]["adapted_only"]["avg_acc"],
        )
        self.assertEqual(
            res_fr["variants_by_name"]["unnormalized_concat"]["avg_acc"],
            res_fr["variants_by_name"]["ptm_only"]["avg_acc"],
        )

        # Verify delta calculation against unnormalized_concat
        unnorm = res_base["variants_by_name"]["unnormalized_concat"]
        norm05 = res_base["variants_by_name"]["normalized_alpha_0.5"]
        self.assertAlmostEqual(
            norm05["delta_avg_acc_vs_unnorm"],
            round(norm05["avg_acc"] - unnorm["avg_acc"], 2),
            places=2,
        )

    def test_benchmark_inference_variants_validation_and_mock_execution(self):
        frozen = _TinyConvBackbone(out_dim=32, seed=1)
        adapted = _TinyConvBackbone(out_dim=32, seed=2)

        for bad_kwargs in (
            {"batch_size": 0},
            {"batch_size": -4},
            {"warmup_iters": 0},
            {"timed_iters": -1},
            {"pca_dim": 0},
            {"num_classes": 0},
            {"batch_size": True},
        ):
            with self.assertRaises(ValueError):
                benchmark_inference_variants(
                    frozen,
                    adapted,
                    input_shape=(3, 16, 16),
                    device="cpu",
                    **bad_kwargs,
                )

        res = benchmark_inference_variants(
            frozen,
            adapted,
            batch_size=4,
            input_shape=(3, 16, 16),
            num_classes=20,
            pca_dim=8,
            warmup_iters=1,
            timed_iters=3,
            device="cpu",
        )
        self.assertEqual(res["memory_backend"], "tracemalloc_mb")
        self.assertEqual([v["variant"] for v in res["variants"]], ["simplecil", "aper_full", "aper_pca"])

        simple = res["variants_by_name"]["simplecil"]
        aper_full = res["variants_by_name"]["aper_full"]
        aper_pca = res["variants_by_name"]["aper_pca"]

        self.assertEqual(simple["branches"], 1)
        self.assertEqual(simple["feat_dim"], 32)
        self.assertEqual(aper_full["branches"], 2)
        self.assertEqual(aper_full["feat_dim"], 64)
        self.assertEqual(aper_pca["branches"], 2)
        self.assertEqual(aper_pca["feat_dim"], 8)

        self.assertAlmostEqual(simple["latency_ratio_vs_simplecil"], 1.0, places=6)
        self.assertAlmostEqual(simple["params_ratio_vs_simplecil"], 1.0, places=6)
        self.assertAlmostEqual(aper_full["params_ratio_vs_simplecil"], 2.0, places=2)
        self.assertAlmostEqual(
            aper_pca["prototype_footprint_mb"] / aper_full["prototype_footprint_mb"],
            8.0 / 64.0,
            places=6,
        )
        self.assertGreater(aper_full["images_per_sec"], 0.0)
        self.assertGreaterEqual(aper_full["peak_memory_mb"], 0.0)

    def test_exports_pillow_plots_and_wandb_empirics_tagging(self):
        cache = _make_synthetic_cache(adapted_scale=2.5, frozen_scale=1.0)
        audit = run_norm_merge_audit(cache, best_alpha=0.5)
        frozen = _TinyConvBackbone(out_dim=16, seed=10)
        adapted = _TinyConvBackbone(out_dim=16, seed=11)
        lat_res = benchmark_inference_variants(
            frozen,
            adapted,
            batch_size=2,
            input_shape=(3, 16, 16),
            num_classes=8,
            pca_dim=4,
            warmup_iters=1,
            timed_iters=2,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_paths = export_critique_csvs(tmpdir, norm_audits=[audit], latency_result=lat_res)
            self.assertIn("norm_imbalance_summary", csv_paths)
            self.assertIn("norm_merge_comparison", csv_paths)
            self.assertIn("inference_latency_benchmark", csv_paths)

            with open(csv_paths["norm_merge_comparison"], "r", encoding="utf-8") as f:
                merge_rows = list(csv.DictReader(f))
            self.assertEqual(len(merge_rows), 5)

            with open(csv_paths["inference_latency_benchmark"], "r", encoding="utf-8") as f:
                lat_rows = list(csv.DictReader(f))
            self.assertEqual(len(lat_rows), 3)

            json_path = export_critique_json(tmpdir, norm_audits=[audit], latency_result=lat_res)
            with open(json_path, "r", encoding="utf-8") as f:
                json_payload = json.load(f)
            self.assertEqual(len(json_payload["norm_audits"]), 1)
            self.assertEqual(len(json_payload["latency_benchmark"]["variants"]), 3)

            # Force Pillow fallback even if matplotlib happens to be installed
            with patch.dict(sys.modules, {"matplotlib": None}):
                plot_paths = generate_critique_plots(tmpdir, norm_audits=[audit], latency_result=lat_res)

            self.assertIn("norm_imbalance_audit", plot_paths)
            self.assertIn("inference_latency_memory", plot_paths)
            for p_path in plot_paths.values():
                self.assertTrue(os.path.exists(p_path))
                with open(p_path, "rb") as img_f:
                    self.assertEqual(img_f.read(4), b"\x89PNG")

            # Disabled W&B should be a strict no-op without importing wandb
            self.assertIsNone(
                log_critique_to_wandb(
                    {"use_wandb": False},
                    norm_audits=[audit],
                    latency_result=lat_res,
                    plot_paths=plot_paths,
                    csv_paths=csv_paths,
                    json_path=json_path,
                )
            )

            # Mock W&B and verify entity, project, mandatory empirics tags, tables, and images
            fake_run = MagicMock()
            fake_run.summary = {}
            fake_wandb = types.SimpleNamespace(
                run=fake_run,
                init=MagicMock(return_value=fake_run),
                Table=MagicMock(side_effect=lambda columns, data: {"columns": columns, "data": data}),
                Image=MagicMock(side_effect=lambda path: f"wandb_image:{os.path.basename(path)}"),
                log=MagicMock(),
                finish=MagicMock(),
            )
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                log_critique_to_wandb(
                    {
                        "wandb": True,
                        "mode": "all",
                        "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
                        "wandb_tags": ["poster-a2"],
                    },
                    norm_audits=[audit],
                    latency_result=lat_res,
                    plot_paths=plot_paths,
                    csv_paths=csv_paths,
                    json_path=json_path,
                )

            fake_wandb.init.assert_called_once()
            init_kwargs = fake_wandb.init.call_args.kwargs
            self.assertEqual(init_kwargs["entity"], "nnm-hcmus")
            self.assertEqual(init_kwargs["project"], "Group7-ML-Lab")
            for required_tag in DEFAULT_CRITIQUE_WANDB_TAGS:
                self.assertIn(required_tag, init_kwargs["tags"])
            self.assertIn("poster-a2", init_kwargs["tags"])
            self.assertIn("backbone:pretrained_vit_b16_224_in21k_adapter", init_kwargs["tags"])

            fake_wandb.log.assert_called_once()
            logged_dict = fake_wandb.log.call_args.args[0]
            self.assertIn("critique/norm_imbalance_table", logged_dict)
            self.assertIn("critique/norm_merge_table", logged_dict)
            self.assertIn("critique/inference_latency_table", logged_dict)
            self.assertIn("critique/norm_imbalance_audit", logged_dict)
            self.assertIn("critique/inference_latency_memory", logged_dict)
            self.assertIn("norm_audit/test_mean_alpha_eff", fake_run.summary)
            self.assertIn("latency_bench/aper_full_latency_ratio", fake_run.summary)
            fake_wandb.finish.assert_called_once()

    def test_cli_help_and_offline_mock_execution(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        help_proc = subprocess.run(
            [sys.executable, "run_critique_micro_exps.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_proc.returncode, 0, msg=help_proc.stderr)
        self.assertIn("--mode", help_proc.stdout)
        self.assertIn("--backbone_source", help_proc.stdout)

        with self.assertRaises(ValueError):
            run_critique_micro_exps.run_critique_micro_experiments({"mode": "norm_audit"})

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "temp_cache.pt")
            save_feature_cache(_make_synthetic_cache(adapted_scale=3.0, frozen_scale=1.0), cache_file)
            out_dir = os.path.join(tmpdir, "out")

            cli_proc = subprocess.run(
                [
                    sys.executable,
                    "run_critique_micro_exps.py",
                    "--mode",
                    "all",
                    "--cache_path",
                    cache_file,
                    "--output_dir",
                    out_dir,
                    "--backbone_source",
                    "mock",
                    "--batch_size",
                    "2",
                    "--num_classes",
                    "10",
                    "--pca_dim",
                    "64",
                    "--warmup_iters",
                    "1",
                    "--timed_iters",
                    "2",
                    "--device",
                    "cpu",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(cli_proc.returncode, 0, msg=cli_proc.stderr)
            for fname in (
                "norm_imbalance_summary.csv",
                "norm_merge_comparison.csv",
                "inference_latency_benchmark.csv",
                "critique_audit_summary.json",
                "norm_imbalance_audit.png",
                "inference_latency_memory.png",
            ):
                self.assertTrue(os.path.exists(os.path.join(out_dir, fname)), msg=f"Missing {fname}")


if __name__ == "__main__":
    unittest.main()
