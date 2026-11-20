import csv
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analysis import sensitivity_sweep
from analysis.sensitivity_sweep import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_K_GRID,
    build_weighted_features,
    evaluate_cil_trajectory,
    export_sweep_csv,
    extract_and_cache_features,
    fit_stage1_pca,
    generate_sensitivity_plots,
    load_feature_cache,
    log_sensitivity_to_wandb,
    run_2d_sensitivity_sweep,
    save_feature_cache,
)
from models.classifiers import CosinePrototypeClassifier
from models.merged_encoder import DualBranchEncoder
import run_exp3_sensitivity


class _DummyProjBackbone(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(out_dim, in_dim, generator=g))
        self.out_dim = out_dim

    def forward(self, x):
        return self.proj(x)


class _DummyTaskManager:
    def __init__(self, xs_train, ys_train, xs_test, ys_test, increments):
        self.xs_train = xs_train
        self.ys_train = ys_train
        self.xs_test = xs_test
        self.ys_test = ys_test
        self._increments = list(increments)
        self.nb_tasks = len(self._increments)
        self.loader_calls = []

    def get_class_ids(self, task_id: int):
        start = sum(self._increments[:task_id])
        end = start + self._increments[task_id]
        return list(range(start, end))

    def get_task_loaders(self, task_id: int):
        self.loader_calls.append(task_id)
        start = sum(self._increments[:task_id])
        end = start + self._increments[task_id]
        tr_mask = (self.ys_train >= start) & (self.ys_train < end)
        te_mask = self.ys_test < end

        tr_x, tr_y = self.xs_train[tr_mask], self.ys_train[tr_mask]
        te_x, te_y = self.xs_test[te_mask], self.ys_test[te_mask]

        train_ds = TensorDataset(torch.arange(len(tr_y)), tr_x, tr_y)
        test_ds = TensorDataset(torch.arange(len(te_y)), te_x, te_y)
        proto_ds = TensorDataset(torch.arange(len(tr_y)), tr_x, tr_y)

        return (
            DataLoader(train_ds, batch_size=8, shuffle=False),
            DataLoader(test_ds, batch_size=8, shuffle=False),
            DataLoader(proto_ds, batch_size=8, shuffle=False),
        )


class _DummyLearner:
    def __init__(self, in_dim: int = 8, feat_dim: int = 16):
        self.device = torch.device("cpu")
        self.args = {
            "dataset": "cifar224",
            "model_name": "aper_adapter",
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
            "seed": 1993,
            "init_cls": 2,
            "increment": 2,
            "adapt_stages": 1,
        }
        self.frozen_ptm = _DummyProjBackbone(in_dim, feat_dim, seed=1)
        for p in self.frozen_ptm.parameters():
            p.requires_grad = False
        self.adapted_ptm = _DummyProjBackbone(in_dim, feat_dim, seed=2)
        self.adapt_calls = []

    def _adapt_model(self, train_loader, nb_old=0, nb_new=None, task_id=0):
        self.adapt_calls.append((task_id, nb_old, nb_new))
        with torch.no_grad():
            self.adapted_ptm.proj.weight.add_(0.25)


class TestExp3FeatureCaching(unittest.TestCase):
    def _build_synthetic_task_data(self, num_classes=6, per_class_train=4, per_class_test=3, in_dim=8):
        g = torch.Generator().manual_seed(123)
        xs_train = torch.randn(num_classes * per_class_train, in_dim, generator=g)
        ys_train = torch.arange(num_classes).repeat_interleave(per_class_train)
        xs_test = torch.randn(num_classes * per_class_test, in_dim, generator=g)
        ys_test = torch.arange(num_classes).repeat_interleave(per_class_test)
        return xs_train, ys_train, xs_test, ys_test

    def test_extract_and_cache_features_round_trip_and_reuse(self):
        xs_train, ys_train, xs_test, ys_test = self._build_synthetic_task_data()
        tm = _DummyTaskManager(xs_train, ys_train, xs_test, ys_test, increments=[2, 2, 2])
        learner = _DummyLearner(in_dim=8, feat_dim=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "nested", "cache.pt")
            cached = extract_and_cache_features(learner, tm, cache_path=cache_path)

            self.assertTrue(os.path.exists(cache_path))
            self.assertEqual(learner.adapt_calls, [(0, 0, 2)])
            self.assertEqual(tm.loader_calls, [0, 1, 2])
            self.assertTrue(all(not p.requires_grad for p in learner.adapted_ptm.parameters()))
            self.assertTrue(all(not p.requires_grad for p in learner.frozen_ptm.parameters()))

            self.assertEqual(cached["train_adapted"].shape, (24, 16))
            self.assertEqual(cached["train_frozen"].shape, (24, 16))
            self.assertEqual(cached["test_adapted"].shape, (18, 16))
            self.assertEqual(cached["test_frozen"].shape, (18, 16))
            self.assertEqual(cached["increments"], [2, 2, 2])
            self.assertEqual(cached["stage0_classes"], [0, 1])
            self.assertEqual(cached["feature_dim"], 16)
            self.assertEqual(cached["merged_dim"], 32)
            self.assertGreater(cached["metadata"]["trainable_params"], 0)

            loaded = load_feature_cache(cache_path)
            self.assertTrue(np.allclose(loaded["train_adapted"], cached["train_adapted"]))
            self.assertTrue(np.allclose(loaded["test_frozen"], cached["test_frozen"]))
            self.assertTrue(np.array_equal(loaded["train_targets"], cached["train_targets"]))

            # Reuse existing cache without adapting or invoking DataLoaders
            tm_reused = _DummyTaskManager(xs_train, ys_train, xs_test, ys_test, increments=[2, 2, 2])
            learner_reused = _DummyLearner(in_dim=8, feat_dim=16)
            reused = extract_and_cache_features(learner_reused, tm_reused, cache_path=cache_path)
            self.assertEqual(learner_reused.adapt_calls, [])
            self.assertEqual(tm_reused.loader_calls, [])
            self.assertTrue(np.allclose(reused["train_adapted"], cached["train_adapted"]))

            # Direct load works even without learner/task_manager when cache file exists
            reused_no_learner = extract_and_cache_features(None, None, cache_path=cache_path)
            self.assertTrue(np.allclose(reused_no_learner["test_adapted"], cached["test_adapted"]))

    def test_save_feature_cache_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "feat.pt")
            payload = {
                "train_adapted": np.zeros((4, 8), dtype=np.float32),
                "train_frozen": np.ones((4, 8), dtype=np.float32),
                "train_targets": np.array([0, 0, 1, 1], dtype=np.int64),
                "test_adapted": np.zeros((2, 8), dtype=np.float32),
                "test_frozen": np.ones((2, 8), dtype=np.float32),
                "test_targets": np.array([0, 1], dtype=np.int64),
                "increments": [1, 1],
            }
            saved = save_feature_cache(payload, path)
            self.assertTrue(os.path.exists(saved))
            loaded = load_feature_cache(saved)
            self.assertEqual(loaded["merged_dim"], 16)
            self.assertEqual(loaded["stage0_classes"], [0])

    def test_extract_and_cache_features_persists_training_hyperparams_in_metadata(self):
        xs_train, ys_train, xs_test, ys_test = self._build_synthetic_task_data()
        tm = _DummyTaskManager(xs_train, ys_train, xs_test, ys_test, increments=[2, 2, 2])
        learner = _DummyLearner(in_dim=8, feat_dim=16)
        learner.args.update(
            {
                "tuned_epoch": 20,
                "init_lr": 0.01,
                "batch_size": 48,
                "weight_decay": 0.0005,
                "ffn_num": 64,
                "optimizer": "sgd",
                "adapt_stages": 1,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "hyper_cache.pt")
            cached = extract_and_cache_features(learner, tm, cache_path=cache_path)
            reloaded = load_feature_cache(cache_path)

            for data in (cached, reloaded):
                meta = data["metadata"]
                self.assertEqual(meta["tuned_epoch"], 20)
                self.assertAlmostEqual(meta["init_lr"], 0.01)
                self.assertEqual(meta["batch_size"], 48)
                self.assertAlmostEqual(meta["weight_decay"], 0.0005)
                self.assertEqual(meta["ffn_num"], 64)
                self.assertEqual(meta["optimizer"], "sgd")
                self.assertEqual(meta["adapt_stages"], 1)

    def test_check_cache_compatibility_detects_mismatches_and_allows_matching(self):
        cached = {
            "metadata": {
                "dataset": "cifar224",
                "model_name": "aper_adapter",
                "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
                "seed": 1993,
                "init_cls": 5,
                "increment": 5,
                "tuned_epoch": 20,
                "init_lr": 0.01,
                "batch_size": 48,
                "weight_decay": 0.0005,
                "ffn_num": 64,
                "optimizer": "sgd",
                "adapt_stages": 1,
            }
        }
        matching_args = {
            "dataset": "cifar224",
            "model_name": "aper_adapter",
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
            "seed": [1993],
            "init_cls": 5,
            "increment": 5,
            "tuned_epoch": 20,
            "init_lr": 0.01,
            "batch_size": 48,
            "weight_decay": 0.0005,
            "ffn_num": 64,
            "optimizer": "sgd",
            "adapt_stages": 1,
        }
        compatible, mismatches = sensitivity_sweep.check_cache_compatibility(cached, matching_args)
        self.assertTrue(compatible)
        self.assertEqual(mismatches, [])

        legacy_cached = {
            "metadata": {
                "dataset": "cifar224",
                "model_name": "aper_adapter",
                "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
                "seed": 1993,
                "init_cls": 5,
                "increment": 5,
            }
        }
        legacy_ok, legacy_mismatches = sensitivity_sweep.check_cache_compatibility(legacy_cached, matching_args)
        self.assertTrue(legacy_ok)
        self.assertEqual(legacy_mismatches, [])

        mismatch_cases = [
            ("tuned_epoch", 1),
            ("init_lr", 0.05),
            ("batch_size", 24),
            ("weight_decay", 0.005),
            ("ffn_num", 32),
            ("dataset", "imagenetr"),
            ("convnet_type", "pretrained_vit_b16_224_in21k_vpt"),
            ("seed", 1994),
            ("init_cls", 10),
            ("increment", 10),
            ("adapt_stages", 0),
        ]
        for key, bad_val in mismatch_cases:
            bad_args = dict(matching_args)
            bad_args[key] = bad_val
            if key == "convnet_type":
                bad_args.pop("backbone_type", None)
            ok, reasons = sensitivity_sweep.check_cache_compatibility(cached, bad_args)
            self.assertFalse(ok, f"Expected mismatch when {key}={bad_val}")
            self.assertTrue(len(reasons) >= 1)

    def test_extract_and_cache_features_reextracts_on_hyperparam_mismatch(self):
        xs_train, ys_train, xs_test, ys_test = self._build_synthetic_task_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "stale_then_fresh.pt")
            tm_a = _DummyTaskManager(xs_train, ys_train, xs_test, ys_test, increments=[2, 2, 2])
            learner_a = _DummyLearner(in_dim=8, feat_dim=16)
            learner_a.args.update({"tuned_epoch": 1, "init_lr": 0.05})
            extract_and_cache_features(learner_a, tm_a, cache_path=cache_path)
            self.assertEqual(learner_a.adapt_calls, [(0, 0, 2)])

            tm_b = _DummyTaskManager(xs_train, ys_train, xs_test, ys_test, increments=[2, 2, 2])
            learner_b = _DummyLearner(in_dim=8, feat_dim=16)
            learner_b.args.update({"tuned_epoch": 20, "init_lr": 0.01})
            fresh = extract_and_cache_features(learner_b, tm_b, cache_path=cache_path, force_reextract=False)

            self.assertEqual(learner_b.adapt_calls, [(0, 0, 2)])
            self.assertEqual(tm_b.loader_calls, [0, 1, 2])
            reloaded = load_feature_cache(cache_path)
            self.assertEqual(reloaded["metadata"]["tuned_epoch"], 20)
            self.assertAlmostEqual(reloaded["metadata"]["init_lr"], 0.01)
            self.assertEqual(fresh["metadata"]["tuned_epoch"], 20)


class TestExp3SensitivitySweep(unittest.TestCase):
    def _build_separable_cached_data(
        self,
        num_classes=6,
        increments=(2, 2, 2),
        feat_dim=32,
        train_per_cls=10,
        test_per_cls=6,
        seed=77,
    ):
        rng = np.random.default_rng(seed)
        centers_adapted = rng.normal(0.0, 1.0, size=(num_classes, feat_dim)).astype(np.float32)
        centers_frozen = rng.normal(0.0, 1.0, size=(num_classes, feat_dim)).astype(np.float32)

        tr_ad, tr_fr, tr_y = [], [], []
        te_ad, te_fr, te_y = [], [], []
        for c in range(num_classes):
            tr_ad.append(centers_adapted[c] + rng.normal(0.0, 0.15, size=(train_per_cls, feat_dim)))
            tr_fr.append(centers_frozen[c] + rng.normal(0.0, 0.35, size=(train_per_cls, feat_dim)))
            tr_y.append(np.full(train_per_cls, c, dtype=np.int64))

            te_ad.append(centers_adapted[c] + rng.normal(0.0, 0.15, size=(test_per_cls, feat_dim)))
            te_fr.append(centers_frozen[c] + rng.normal(0.0, 0.35, size=(test_per_cls, feat_dim)))
            te_y.append(np.full(test_per_cls, c, dtype=np.int64))

        return {
            "train_adapted": np.concatenate(tr_ad, axis=0).astype(np.float32),
            "train_frozen": np.concatenate(tr_fr, axis=0).astype(np.float32),
            "train_targets": np.concatenate(tr_y, axis=0),
            "test_adapted": np.concatenate(te_ad, axis=0).astype(np.float32),
            "test_frozen": np.concatenate(te_fr, axis=0).astype(np.float32),
            "test_targets": np.concatenate(te_y, axis=0),
            "increments": list(increments),
            "stage0_classes": list(range(increments[0])),
            "feature_dim": feat_dim,
            "merged_dim": feat_dim * 2,
            "metadata": {
                "dataset": "cifar224",
                "model_name": "aper_adapter",
                "method_name": "aper_adapter",
                "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
                "seed": 1993,
                "init_cls": increments[0],
                "increment": increments[1],
                "total_params": 1000,
                "trainable_params": 100,
            },
        }

    def test_exact_cosine_equivalence_with_dual_branch_encoder_and_svd_pca(self):
        torch.manual_seed(0)
        frozen = _DummyProjBackbone(in_dim=8, out_dim=16, seed=101)
        adapted = _DummyProjBackbone(in_dim=8, out_dim=16, seed=202)
        xs_train = torch.randn(24, 8)
        ys_train = torch.arange(6).repeat_interleave(4)
        xs_test = torch.randn(18, 8)
        ys_test = torch.arange(6).repeat_interleave(3)
        increments = [2, 2, 2]

        with torch.no_grad():
            tr_ad = adapted(xs_train).numpy().astype(np.float32)
            tr_fr = frozen(xs_train).numpy().astype(np.float32)
            te_ad = adapted(xs_test).numpy().astype(np.float32)
            te_fr = frozen(xs_test).numpy().astype(np.float32)

        for alpha in (0.0, 0.3, 0.5, 1.0):
            # Ground-truth PyTorch DualBranchEncoder + CosinePrototypeClassifier trajectory
            enc = DualBranchEncoder(frozen, adapted, merge_mode="both", alpha=alpha, weighted_norm=True)
            clf = CosinePrototypeClassifier(in_features=enc.out_dim)
            pt_top1_curve = []
            cursor = 0
            with torch.no_grad():
                tr_feat_pt = enc(xs_train)
                te_feat_pt = enc(xs_test)
                for inc in increments:
                    cids = list(range(cursor, cursor + inc))
                    protos = torch.stack([tr_feat_pt[ys_train == c].mean(dim=0) for c in cids], dim=0)
                    clf.register_prototypes(protos)
                    cursor += inc
                    mask = ys_test < cursor
                    preds = clf(te_feat_pt[mask]).argmax(dim=1)
                    pt_top1_curve.append(round(float((preds == ys_test[mask]).float().mean().item() * 100.0), 2))

            cached = {
                "train_adapted": tr_ad,
                "train_frozen": tr_fr,
                "train_targets": ys_train.numpy(),
                "test_adapted": te_ad,
                "test_frozen": te_fr,
                "test_targets": ys_test.numpy(),
                "increments": increments,
                "stage0_classes": [0, 1],
            }
            res = run_2d_sensitivity_sweep(cached, alpha_grid=[alpha], k_grid=[32], method="pca")
            row = res["rows"][0]
            self.assertEqual(row["effective_k"], 32)
            self.assertFalse(row["k_capped"])
            self.assertEqual(row["top1_curve"], pt_top1_curve)
            self.assertAlmostEqual(row["avg_acc"], round(float(np.mean(pt_top1_curve)), 2), places=2)

        # Verify numpy_svd fallback produces orthonormal components summing <= 1 EVR
        tr_norm_ad = tr_ad / np.maximum(np.linalg.norm(tr_ad, axis=-1, keepdims=True), 1e-12)
        tr_norm_fr = tr_fr / np.maximum(np.linalg.norm(tr_fr, axis=-1, keepdims=True), 1e-12)
        phi = build_weighted_features(tr_norm_ad, tr_norm_fr, alpha=0.5)
        mean, comps, evr, backend = fit_stage1_pca(phi[:8], max_components=6, prefer_sklearn=False)
        self.assertEqual(backend, "numpy_svd")
        self.assertEqual(mean.shape, (32,))
        self.assertEqual(comps.shape, (6, 32))
        self.assertTrue(np.allclose(comps @ comps.T, np.eye(6), atol=1e-5))
        self.assertLessEqual(float(np.sum(evr)), 1.0 + 1e-5)

    def test_pca_rank_cap_safety_on_small_stage1_dataset(self):
        # Stage 1 has 2 classes x 10 samples = 20 samples < requested k=32 < merged_dim=64
        cached = self._build_separable_cached_data(
            num_classes=6, increments=(2, 2, 2), feat_dim=32, train_per_cls=10, test_per_cls=5
        )
        res = run_2d_sensitivity_sweep(
            cached,
            alpha_grid=[0.5],
            k_grid=[16, 32, 64],
            method="both",
            random_seeds=[1993, 1994, 1995],
        )
        self.assertEqual(res["n_stage0_samples"], 20)
        self.assertEqual(res["max_pca_rank"], 20)
        self.assertEqual(len(res["rows"]), 3)

        by_k = {r["k"]: r for r in res["rows"]}
        self.assertEqual(by_k[16]["effective_k"], 16)
        self.assertFalse(by_k[16]["k_capped"])

        self.assertEqual(by_k[32]["effective_k"], 20)
        self.assertTrue(by_k[32]["k_capped"])
        self.assertIsNotNone(by_k[32]["pca_avg_acc"])
        self.assertIsNotNone(by_k[32]["random_avg_acc"])

        # k = 2d bypasses PCA completely
        self.assertEqual(by_k[64]["effective_k"], 64)
        self.assertFalse(by_k[64]["k_capped"])
        self.assertEqual(by_k[64]["explained_variance_ratio"], 1.0)
        self.assertEqual(by_k[64]["random_avg_acc_std"], 0.0)

    def test_full_11x8_grid_and_multiseed_random_subsampling(self):
        cached = self._build_separable_cached_data(
            num_classes=10, increments=(5, 5), feat_dim=768, train_per_cls=12, test_per_cls=4
        )
        res = run_2d_sensitivity_sweep(
            cached,
            alpha_grid=DEFAULT_ALPHA_GRID,
            k_grid=DEFAULT_K_GRID,
            method="both",
            random_seeds=[1993, 1994, 1995],
        )
        self.assertEqual(len(res["alpha_grid"]), 11)
        self.assertEqual(len(res["k_grid"]), 8)
        self.assertEqual(len(res["rows"]), 88)

        required_fields = {
            "alpha",
            "k",
            "effective_k",
            "k_capped",
            "avg_acc",
            "last_acc",
            "old_acc",
            "new_acc",
            "pca_avg_acc",
            "pca_last_acc",
            "random_avg_acc",
            "random_avg_acc_std",
            "random_last_acc",
            "random_last_acc_std",
        }
        for row in res["rows"]:
            self.assertTrue(required_fields.issubset(row.keys()))
            self.assertIsNotNone(row["avg_acc"])
            self.assertIsNotNone(row["last_acc"])
            self.assertIsNotNone(row["old_acc"])
            self.assertIsNotNone(row["new_acc"])
            self.assertIsNotNone(row["random_avg_acc"])
            self.assertIsNotNone(row["random_avg_acc_std"])

    def test_prepare_plot_data_includes_quick_read_insights_and_capped_labels(self):
        cached = self._build_separable_cached_data(
            num_classes=6, increments=(2, 2, 2), feat_dim=16, train_per_cls=5, test_per_cls=4
        )
        sweep_res = run_2d_sensitivity_sweep(
            cached,
            alpha_grid=[0.0, 0.5, 1.0],
            k_grid=[4, 8, 16, 32],
            method="both",
            random_seeds=[1993, 1994],
        )
        plot_data = sensitivity_sweep._prepare_plot_data(sweep_res)

        self.assertTrue(plot_data["any_k_capped"])
        self.assertEqual(plot_data["max_pca_rank"], 10)
        self.assertEqual(plot_data["n_stage0_samples"], 10)
        self.assertIn("16(10)", plot_data["comp_k_labels"])
        self.assertEqual(plot_data["comp_k_labels"], ["4", "8", "16(10)", "32"])

        best_idx = plot_data["best_cell_idx"]
        self.assertIsNotNone(best_idx)
        best = sweep_res["summary"]["best_overall"]
        self.assertEqual(plot_data["alphas"][best_idx[0]], best["alpha"])
        self.assertEqual(plot_data["ks"][best_idx[1]], best["k"])

        default_idx = plot_data["default_cell_idx"]
        self.assertIsNotNone(default_idx)
        self.assertEqual(plot_data["alphas"][default_idx[0]], 0.5)
        self.assertEqual(plot_data["ks"][default_idx[1]], 32)

        self.assertIsInstance(plot_data["delta_best_vs_default_full"], float)
        self.assertIsInstance(plot_data["delta_best_overall_vs_default"], float)
        self.assertIsNotNone(plot_data["best_pareto_128"])

    def test_matplotlib_plots_fix_overlap_show_capped_k_and_annotate_key_points(self):
        cached = self._build_separable_cached_data(
            num_classes=6, increments=(2, 2, 2), feat_dim=16, train_per_cls=5, test_per_cls=4
        )
        sweep_res = run_2d_sensitivity_sweep(
            cached,
            alpha_grid=[0.0, 0.5, 1.0],
            k_grid=[4, 8, 16, 32],
            method="both",
            random_seeds=[1993, 1994],
        )
        plot_data = sensitivity_sweep._prepare_plot_data(sweep_res)

        created_figs = []

        class FakeRectangle:
            def __init__(self, xy, width, height, **kwargs):
                self.xy = xy
                self.width = width
                self.height = height
                self.kwargs = kwargs

        class FakeAxes:
            def __init__(self):
                self.plots = []
                self.vlines = []
                self.hlines = []
                self.fills = []
                self.annotations = []
                self.imshows = []
                self.texts = []
                self.patches = []
                self.xticks = []
                self.xticklabels = []
                self.yticks = []
                self.yticklabels = []
                self.ylims = []
                self.xscales = []
                self.xlabel = ""
                self.ylabel = ""
                self.title = ""

            def plot(self, *args, **kwargs):
                self.plots.append((args, kwargs))

            def axvline(self, x, **kwargs):
                self.vlines.append((x, kwargs))

            def axhline(self, y, **kwargs):
                self.hlines.append((y, kwargs))

            def fill_between(self, *args, **kwargs):
                self.fills.append((args, kwargs))

            def annotate(self, text, xy, **kwargs):
                self.annotations.append((text, xy, kwargs))

            def imshow(self, mat, **kwargs):
                self.imshows.append((mat, kwargs))
                return MagicMock()

            def text(self, x, y, s, **kwargs):
                self.texts.append((x, y, s, kwargs))

            def add_patch(self, patch):
                self.patches.append(patch)

            def set_xticks(self, ticks):
                self.xticks = list(ticks)

            def set_xticklabels(self, labels, **kwargs):
                self.xticklabels = list(labels)

            def set_yticks(self, ticks):
                self.yticks = list(ticks)

            def set_yticklabels(self, labels, **kwargs):
                self.yticklabels = list(labels)

            def set_ylim(self, *args, **kwargs):
                self.ylims.append((args, kwargs))

            def set_xscale(self, *args, **kwargs):
                self.xscales.append((args, kwargs))

            def set_xlabel(self, label, **kwargs):
                self.xlabel = label

            def set_ylabel(self, label, **kwargs):
                self.ylabel = label

            def set_title(self, title, **kwargs):
                self.title = title

            def grid(self, *args, **kwargs):
                pass

            def legend(self, *args, **kwargs):
                pass

        class FakeFigure:
            def __init__(self, axes):
                self.axes = axes
                self.suptitle_calls = []
                self.tight_layout_calls = []
                self.saved_paths = []
                self.colorbars = []

            def suptitle(self, text, **kwargs):
                self.suptitle_calls.append((text, kwargs))

            def tight_layout(self, **kwargs):
                self.tight_layout_calls.append(kwargs)

            def colorbar(self, im, **kwargs):
                self.colorbars.append((im, kwargs))

            def savefig(self, path, **kwargs):
                self.saved_paths.append((path, kwargs))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)

        def fake_subplots(nrows=1, ncols=1, **kwargs):
            if nrows == 1 and ncols == 1:
                ax = FakeAxes()
                fig = FakeFigure(ax)
                created_figs.append(fig)
                return fig, ax
            axes = [FakeAxes() for _ in range(ncols)]
            fig = FakeFigure(axes)
            created_figs.append(fig)
            return fig, axes

        fake_mpl = types.ModuleType("matplotlib")
        fake_mpl.use = MagicMock()
        fake_plt = types.ModuleType("matplotlib.pyplot")
        fake_plt.subplots = fake_subplots
        fake_plt.close = MagicMock()
        fake_patches = types.ModuleType("matplotlib.patches")
        fake_patches.Rectangle = FakeRectangle

        saved_modules = {name: sys.modules.get(name) for name in ("matplotlib", "matplotlib.pyplot", "matplotlib.patches")}
        sys.modules["matplotlib"] = fake_mpl
        sys.modules["matplotlib.pyplot"] = fake_plt
        sys.modules["matplotlib.patches"] = fake_patches
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = generate_sensitivity_plots(sweep_res, tmpdir)
                for key in ("alpha_curve", "pca_compression_curve", "heatmap_alpha_k"):
                    self.assertTrue(os.path.exists(paths[key]))
        finally:
            for name, orig in saved_modules.items():
                if orig is not None:
                    sys.modules[name] = orig
                else:
                    sys.modules.pop(name, None)

        self.assertEqual(len(created_figs), 3)

        # Fig 1: alpha_curve.png
        fig_alpha = created_figs[0]
        ax_alpha = fig_alpha.axes
        self.assertTrue(any(kw.get("marker") == "*" for _, kw in ax_alpha.plots))
        self.assertTrue(any(abs(x - 0.5) < 1e-6 for x, _ in ax_alpha.vlines))
        best_alpha_str = f"{plot_data['best_full_alpha']:.1f}"
        self.assertTrue(
            any("Best" in txt and best_alpha_str in txt for txt, _, _ in ax_alpha.annotations),
            f"Missing Best alpha annotation in {ax_alpha.annotations}",
        )

        # Fig 2: pca_compression_curve.png
        fig_pca = created_figs[1]
        ax_pca = fig_pca.axes
        self.assertIn("16(10)", ax_pca.xticklabels)
        self.assertNotIn("16", ax_pca.xticklabels)
        self.assertTrue(any(abs(x - 10) < 1e-6 for x, _ in ax_pca.vlines))
        self.assertTrue(
            any("Pareto" in txt for txt, _, _ in ax_pca.annotations),
            f"Missing Pareto compression annotation in {ax_pca.annotations}",
        )

        # Fig 3: heatmap_alpha_k.png
        fig_hm = created_figs[2]
        self.assertEqual(len(fig_hm.tight_layout_calls), 1)
        rect = fig_hm.tight_layout_calls[0].get("rect")
        self.assertIsNotNone(rect)
        self.assertLessEqual(rect[3], 0.94)
        self.assertTrue(fig_hm.suptitle_calls)
        suptitle_txt = fig_hm.suptitle_calls[0][0]
        self.assertIn("Best", suptitle_txt)
        self.assertIn("Default APER", suptitle_txt)

        i_best, j_best = plot_data["best_cell_idx"]
        for hm_ax in fig_hm.axes:
            self.assertIn("16(10)", hm_ax.xticklabels)
            self.assertTrue(
                any(
                    isinstance(p, FakeRectangle)
                    and abs(p.xy[0] - (j_best - 0.5)) < 1e-6
                    and abs(p.xy[1] - (i_best - 0.5)) < 1e-6
                    for p in hm_ax.patches
                )
            )


class TestExp3ArtifactsAndWandb(unittest.TestCase):
    def _run_88_sweep(self):
        helper = TestExp3SensitivitySweep()
        cached = helper._build_separable_cached_data(
            num_classes=10, increments=(5, 5), feat_dim=32, train_per_cls=8, test_per_cls=4
        )
        return run_2d_sensitivity_sweep(
            cached,
            alpha_grid=DEFAULT_ALPHA_GRID,
            k_grid=[4, 8, 16, 24, 32, 48, 56, 64],
            method="both",
            random_seeds=[1993, 1994, 1995],
        )

    def test_csv_and_three_png_plots_generated_without_wandb(self):
        sweep_res = self._run_88_sweep()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "exp3_sweep.csv")
            written_csv = export_sweep_csv(sweep_res, csv_path)
            self.assertTrue(os.path.exists(written_csv))

            with open(written_csv, "r", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), 88)
            self.assertEqual(csv_rows[0]["dataset"], "cifar224")

            plot_paths = generate_sensitivity_plots(sweep_res, tmpdir)
            self.assertEqual(
                set(plot_paths.keys()),
                {"alpha_curve", "pca_compression_curve", "heatmap_alpha_k"},
            )
            for key, filename in [
                ("alpha_curve", "alpha_curve.png"),
                ("pca_compression_curve", "pca_compression_curve.png"),
                ("heatmap_alpha_k", "heatmap_alpha_k.png"),
            ]:
                path = plot_paths[key]
                self.assertEqual(os.path.basename(path), filename)
                self.assertTrue(os.path.exists(path))
                with open(path, "rb") as img_f:
                    header = img_f.read(8)
                    rest = img_f.read()
                self.assertEqual(header, b"\x89PNG\r\n\x1a\n")
                self.assertGreater(len(rest), 100)

            self.assertIsNone(log_sensitivity_to_wandb(sweep_res, {"use_wandb": False}, plot_paths, written_csv))

    def test_log_sensitivity_to_wandb_logs_table_figures_tags_and_summary(self):
        sweep_res = self._run_88_sweep()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = export_sweep_csv(sweep_res, os.path.join(tmpdir, "exp3.csv"))
            plot_paths = generate_sensitivity_plots(sweep_res, tmpdir)

            mock_run = MagicMock()
            mock_run.summary = {}
            mock_wandb = types.ModuleType("wandb")
            mock_wandb.run = mock_run
            mock_wandb.init = MagicMock(return_value=mock_run)
            mock_wandb.log = MagicMock()
            mock_wandb.finish = MagicMock()
            mock_wandb.Table = MagicMock(side_effect=lambda columns, data: {"columns": columns, "data": data})
            mock_wandb.Image = MagicMock(side_effect=lambda path: f"image:{path}")

            orig_wandb = sys.modules.get("wandb")
            sys.modules["wandb"] = mock_wandb
            try:
                args = {
                    "use_wandb": True,
                    "dataset": "cifar224",
                    "model_name": "aper_adapter",
                    "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
                    "init_cls": 5,
                    "increment": 5,
                    "seed": 1993,
                    "wandb_tags": ["exp3"],
                }
                log_sensitivity_to_wandb(sweep_res, args, plot_paths=plot_paths, csv_path=csv_path)
            finally:
                if orig_wandb is not None:
                    sys.modules["wandb"] = orig_wandb
                else:
                    sys.modules.pop("wandb", None)

            mock_wandb.init.assert_called_once()
            init_kwargs = mock_wandb.init.call_args.kwargs
            self.assertEqual(init_kwargs["entity"], "nnm-hcmus")
            self.assertEqual(init_kwargs["project"], "Group7-ML-Lab")
            self.assertEqual(init_kwargs["config"]["backbone_type"], "pretrained_vit_b16_224_in21k_adapter")
            self.assertIn("backbone:pretrained_vit_b16_224_in21k_adapter", init_kwargs["tags"])
            self.assertIn("exp3", init_kwargs["tags"])

            table_kwargs = mock_wandb.Table.call_args.kwargs
            self.assertEqual(len(table_kwargs["data"]), 88)
            self.assertEqual(mock_wandb.Image.call_count, 3)

            logged_payload = mock_wandb.log.call_args.args[0]
            self.assertIn("exp3/sensitivity_table", logged_payload)
            self.assertIn("exp3/alpha_curve", logged_payload)
            self.assertIn("exp3/pca_compression_curve", logged_payload)
            self.assertIn("exp3/heatmap_alpha_k", logged_payload)

            best = sweep_res["summary"]["best_overall"]
            self.assertEqual(mock_run.summary["best_alpha"], best["alpha"])
            self.assertEqual(mock_run.summary["best_k"], best["k"])
            self.assertEqual(mock_run.summary["best_avg_acc"], best["avg_acc"])
            self.assertEqual(mock_run.summary["best_last_acc"], best["last_acc"])
            mock_wandb.finish.assert_called_once()


class TestExp3CliAndConfigs(unittest.TestCase):
    def test_exp3_json_configs_parse_cleanly(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        expected_specs = {
            "sweep_cifar224_adapter.json": ("cifar224", 5, 5),
            "sweep_imagenetr_adapter.json": ("imagenetr", 5, 5),
            "sweep_cub_adapter.json": ("cub", 10, 10),
        }
        for filename, (dataset, init_cls, increment) in expected_specs.items():
            cfg_path = os.path.join(repo_root, "exps", "exp3_analysis", filename)
            self.assertTrue(os.path.exists(cfg_path), f"Missing config: {cfg_path}")
            cfg = run_exp3_sensitivity.load_json(cfg_path)
            self.assertEqual(cfg["dataset"], dataset)
            self.assertEqual(cfg["init_cls"], init_cls)
            self.assertEqual(cfg["increment"], increment)
            self.assertEqual(cfg["model_name"], "aper_adapter")
            self.assertEqual(cfg["ffn_num"], 64)
            self.assertEqual(len(cfg["alpha_grid"]), 11)
            self.assertEqual(len(cfg["k_grid"]), 8)

    def test_exp3_configs_match_table1_aper_adapter_hyperparameters(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        pairs = [
            ("exps/exp3_analysis/sweep_cifar224_adapter.json", "exps/aper_adapter/cifar224_5_5_vit-b_adapter.json"),
            ("exps/exp3_analysis/sweep_imagenetr_adapter.json", "exps/aper_adapter/imagenetr_5_5_vit-b_adapter.json"),
            ("exps/exp3_analysis/sweep_cub_adapter.json", "exps/aper_adapter/cub_10_10_vit-b_adapter.json"),
        ]
        shared_keys = (
            "dataset",
            "model_name",
            "convnet_type",
            "init_cls",
            "increment",
            "ffn_num",
            "optimizer",
            "tuned_epoch",
            "init_lr",
            "weight_decay",
            "batch_size",
            "min_lr",
            "seed",
        )
        for sweep_rel, base_rel in pairs:
            sweep_cfg = run_exp3_sensitivity.load_json(os.path.join(repo_root, sweep_rel))
            base_cfg = run_exp3_sensitivity.load_json(os.path.join(repo_root, base_rel))
            for key in shared_keys:
                self.assertEqual(
                    sweep_cfg[key],
                    base_cfg[key],
                    f"{sweep_rel} mismatches {base_rel} on '{key}': {sweep_cfg[key]} != {base_cfg[key]}",
                )

    def test_cli_runs_end_to_end_from_existing_cache(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cfg_path = os.path.join(repo_root, "exps", "exp3_analysis", "sweep_cifar224_adapter.json")
        helper = TestExp3SensitivitySweep()
        cached = helper._build_separable_cached_data(
            num_classes=10, increments=(5, 5), feat_dim=32, train_per_cls=4, test_per_cls=2
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = save_feature_cache(cached, os.path.join(tmpdir, "cached.pt"))
            out_dir = os.path.join(tmpdir, "out")
            argv = [
                "run_exp3_sensitivity.py",
                "--config",
                cfg_path,
                "--cache_path",
                cache_path,
                "--output_dir",
                out_dir,
                "--alpha_grid",
                "0.0",
                "0.5",
                "1.0",
                "--k_grid",
                "16",
                "64",
            ]
            with patch.object(sys, "argv", argv):
                results = run_exp3_sensitivity.main()

            self.assertEqual(len(results), 1)
            res = results[0]
            self.assertEqual(len(res["sweep_result"]["rows"]), 6)
            self.assertTrue(os.path.exists(res["csv_path"]))
            for key in ("alpha_curve", "pca_compression_curve", "heatmap_alpha_k"):
                self.assertTrue(os.path.exists(res["plot_paths"][key]))

    def test_run_experiment_3_reextracts_when_cache_is_stale(self):
        helper = TestExp3SensitivitySweep()
        stale_cached = helper._build_separable_cached_data(
            num_classes=10, increments=(5, 5), feat_dim=32, train_per_cls=4, test_per_cls=2, seed=7
        )
        stale_cached["metadata"].update(
            {
                "dataset": "cifar224",
                "model_name": "aper_adapter",
                "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
                "seed": 1993,
                "init_cls": 5,
                "increment": 5,
                "tuned_epoch": 1,
                "init_lr": 0.01,
                "batch_size": 48,
                "weight_decay": 0.0005,
                "ffn_num": 64,
                "optimizer": "sgd",
                "adapt_stages": 1,
            }
        )
        fresh_cached = helper._build_separable_cached_data(
            num_classes=10, increments=(5, 5), feat_dim=32, train_per_cls=4, test_per_cls=2, seed=99
        )
        fresh_cached["metadata"] = dict(stale_cached["metadata"], tuned_epoch=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = save_feature_cache(stale_cached, os.path.join(tmpdir, "stale_cache.pt"))
            out_dir = os.path.join(tmpdir, "out")
            run_args = dict(stale_cached["metadata"])
            run_args.update(
                {
                    "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
                    "tuned_epoch": 20,
                    "cache_path": cache_path,
                    "output_dir": out_dir,
                    "alpha_grid": [0.0, 0.5, 1.0],
                    "k_grid": [16, 64],
                    "random_seeds": [1993],
                    "device": ["cpu"],
                    "use_wandb": False,
                }
            )

            extract_calls = []

            def fake_extract(learner, task_manager, cache_path=None, force_reextract=False):
                extract_calls.append({"cache_path": cache_path, "force_reextract": force_reextract})
                return fresh_cached

            fake_tm_mod = types.ModuleType("data.task_manager")
            fake_tm_mod.ContinualTaskManager = MagicMock(return_value=MagicMock())
            fake_learner_mod = types.ModuleType("framework.learner")
            fake_learner_mod.APERLearner = MagicMock(return_value=MagicMock())
            fake_learner_mod._set_device = MagicMock()
            fake_learner_mod._set_random = MagicMock()

            with patch.dict(
                sys.modules,
                {"data.task_manager": fake_tm_mod, "framework.learner": fake_learner_mod},
            ), patch("analysis.sensitivity_sweep.extract_and_cache_features", side_effect=fake_extract):
                res = run_exp3_sensitivity.run_experiment_3(run_args)

            self.assertEqual(len(extract_calls), 1)
            self.assertTrue(extract_calls[0]["force_reextract"])
            self.assertEqual(res["sweep_result"]["metadata"]["tuned_epoch"], 20)


if __name__ == "__main__":
    unittest.main()
