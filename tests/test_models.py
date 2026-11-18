import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from framework.learner import APERLearner
from models.merged_encoder import DualBranchEncoder


class _DummyBackbone(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, fill_value: float, return_dict: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.constant_(self.proj.weight, 0.0)
        nn.init.constant_(self.proj.bias, fill_value)
        self.out_dim = out_dim
        self.return_dict = return_dict

    def forward(self, x):
        out = self.proj(x)
        if self.return_dict:
            return {"features": out}
        return out


class TestDualBranchEncoderMergeModes(unittest.TestCase):
    def test_merge_modes_output_dimensions_and_values(self):
        frozen = _DummyBackbone(in_dim=8, out_dim=768, fill_value=1.0, return_dict=True)
        adapted = _DummyBackbone(in_dim=8, out_dim=768, fill_value=2.0, return_dict=False)
        x = torch.randn(3, 8)

        enc_both = DualBranchEncoder(frozen, adapted, merge_mode="both")
        self.assertEqual(enc_both.out_dim, 1536)
        out_both = enc_both(x)
        self.assertEqual(out_both.shape, (3, 1536))
        self.assertTrue(torch.allclose(out_both[:, :768], torch.full((3, 768), 2.0)))
        self.assertTrue(torch.allclose(out_both[:, 768:], torch.full((3, 768), 1.0)))

        enc_adapted = DualBranchEncoder(frozen, adapted, merge_mode="adapted_only")
        self.assertEqual(enc_adapted.out_dim, 768)
        out_adapted = enc_adapted(x)
        self.assertEqual(out_adapted.shape, (3, 768))
        self.assertTrue(torch.allclose(out_adapted, torch.full((3, 768), 2.0)))

        enc_ptm = DualBranchEncoder(frozen, adapted, merge_mode="ptm_only")
        self.assertEqual(enc_ptm.out_dim, 768)
        out_ptm = enc_ptm(x)
        self.assertEqual(out_ptm.shape, (3, 768))
        self.assertTrue(torch.allclose(out_ptm, torch.full((3, 768), 1.0)))

        enc_dual = DualBranchEncoder(frozen, adapted, merge_mode="dual_ptm")
        self.assertEqual(enc_dual.out_dim, 1536)
        out_dual = enc_dual(x)
        self.assertEqual(out_dual.shape, (3, 1536))
        self.assertTrue(torch.allclose(out_dual[:, :768], torch.full((3, 768), 2.0)))
        self.assertTrue(torch.allclose(out_dual[:, 768:], torch.full((3, 768), 1.0)))

    def test_single_branch_modes_allow_none_for_unused_branch(self):
        frozen = _DummyBackbone(in_dim=8, out_dim=768, fill_value=1.0)
        adapted = _DummyBackbone(in_dim=8, out_dim=768, fill_value=2.0)
        x = torch.randn(2, 8)

        enc_ptm = DualBranchEncoder(frozen, None, merge_mode="ptm_only")
        self.assertEqual(enc_ptm(x).shape, (2, 768))

        enc_adapted = DualBranchEncoder(None, adapted, merge_mode="adapted_only")
        self.assertEqual(enc_adapted(x).shape, (2, 768))

    def test_invalid_merge_mode_raises(self):
        frozen = _DummyBackbone(in_dim=8, out_dim=768, fill_value=1.0)
        adapted = _DummyBackbone(in_dim=8, out_dim=768, fill_value=2.0)
        with self.assertRaises(ValueError):
            DualBranchEncoder(frozen, adapted, merge_mode="unknown")


class TestAPERLearnerMergeModes(unittest.TestCase):
    def _make_fake_get_pretrained(self, call_log):
        def fake_get_pretrained(args):
            convnet = args.get("convnet_type")
            call_log.append(convnet)
            fill = 2.0 if convnet == "pretrained_vit_b16_224" else 1.0
            backbone = _DummyBackbone(in_dim=8, out_dim=768, fill_value=fill)
            for p in backbone.parameters():
                p.requires_grad = False
            return backbone

        return fake_get_pretrained

    def _make_loader(self, num_classes=2, start_cls=0):
        xs = torch.randn(num_classes * 2, 8)
        ys = torch.tensor([c for c in range(start_cls, start_cls + num_classes) for _ in range(2)])
        idx = torch.arange(len(ys))
        return DataLoader(TensorDataset(idx, xs, ys), batch_size=4)

    def test_classifier_dims_and_prototype_registration_across_merge_modes(self):
        for mode, expected_dim in [("ptm_only", 768), ("adapted_only", 768), ("both", 1536), ("dual_ptm", 1536)]:
            call_log = []
            args = {
                "device": [torch.device("cpu")],
                "convnet_type": "pretrained_vit_b16_224_in21k",
                "second_convnet_type": "pretrained_vit_b16_224",
                "model_name": "simplecil" if mode != "adapted_only" else "aper_finetune",
                "merge_mode": mode,
                "tuned_epoch": 1,
                "init_cls": 2,
            }
            with patch("framework.learner.get_pretrained_model", side_effect=self._make_fake_get_pretrained(call_log)):
                learner = APERLearner(args)

            self.assertEqual(learner.classifier.in_features, expected_dim)
            self.assertEqual(learner.classifier.weight.shape, (0, expected_dim))
            if mode == "dual_ptm":
                self.assertEqual(call_log, ["pretrained_vit_b16_224_in21k", "pretrained_vit_b16_224"])
                self.assertFalse(learner.strategy.requires_training)
            else:
                self.assertEqual(call_log, ["pretrained_vit_b16_224_in21k"])

            loader = self._make_loader(num_classes=2, start_cls=0)
            learner.execute_task(0, loader, loader, [0, 1])
            self.assertEqual(learner.encoder.out_dim, expected_dim)
            self.assertEqual(learner.classifier.weight.shape, (2, expected_dim))


class _DummyAdapterBackbone(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 8):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.adaptmlp = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        return self.base(x) + self.adaptmlp(self.base(x))


class TestAPERLearnerAdaptStages(unittest.TestCase):
    def _make_loader(self, start_cls: int, num_classes: int = 2, in_dim: int = 8):
        torch.manual_seed(42 + start_cls)
        xs = torch.randn(num_classes * 4, in_dim)
        ys = torch.tensor([c for c in range(start_cls, start_cls + num_classes) for _ in range(4)])
        idx = torch.arange(len(ys))
        return DataLoader(TensorDataset(idx, xs, ys), batch_size=4)

    def _build_adapter_learner(self, adapt_stages: int):
        def fake_get_pretrained(args):
            torch.manual_seed(0)
            backbone = _DummyAdapterBackbone(in_dim=8, out_dim=8)
            for p in backbone.parameters():
                p.requires_grad = False
            return backbone

        args = {
            "device": [torch.device("cpu")],
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
            "model_name": "aper_adapter",
            "merge_mode": "both",
            "adapt_stages": adapt_stages,
            "tuned_epoch": 2,
            "init_lr": 0.1,
            "weight_decay": 0.0,
            "optimizer": "sgd",
            "min_lr": 1e-5,
            "init_cls": 2,
            "increment": 2,
        }
        with patch("framework.learner.get_pretrained_model", side_effect=fake_get_pretrained):
            return APERLearner(args)

    def test_adapt_stages_zero_skips_all_adaptation(self):
        learner = self._build_adapter_learner(adapt_stages=0)
        init_adapter_weight = learner.adapted_ptm.adaptmlp.weight.detach().clone()

        for task_id in range(3):
            loader = self._make_loader(start_cls=task_id * 2, num_classes=2)
            learner.execute_task(task_id, loader, loader, [task_id * 2, task_id * 2 + 1])

        self.assertIsNotNone(learner.encoder)
        self.assertTrue(torch.equal(learner.adapted_ptm.adaptmlp.weight, init_adapter_weight))
        self.assertEqual(learner.classifier.weight.shape, (6, 16))
        self.assertTrue(all(not p.requires_grad for p in learner.adapted_ptm.parameters()))

    def test_multi_stage_adaptation_preserves_w_old_and_freezes_after_T(self):
        learner = self._build_adapter_learner(adapt_stages=2)
        base_weight_init = learner.adapted_ptm.base.weight.detach().clone()
        adapter_init = learner.adapted_ptm.adaptmlp.weight.detach().clone()

        # Stage 0 (< T=2): adapts
        loader0 = self._make_loader(start_cls=0, num_classes=2)
        learner.execute_task(0, loader0, loader0, [0, 1])
        adapter_after_t0 = learner.adapted_ptm.adaptmlp.weight.detach().clone()
        self.assertFalse(torch.equal(adapter_after_t0, adapter_init))
        self.assertTrue(torch.equal(learner.adapted_ptm.base.weight, base_weight_init))
        self.assertTrue(all(not p.requires_grad for p in learner.adapted_ptm.parameters()))
        self.assertEqual(learner.classifier.weight.shape, (2, 16))
        w_old_snapshot = learner.classifier.weight[:2].detach().clone()

        # Stage 1 (1 <= b < T=2): re-unlocks adapter, keeps W_old fixed, updates adapter & W_new
        loader1 = self._make_loader(start_cls=2, num_classes=2)
        learner.execute_task(1, loader1, loader1, [2, 3])
        adapter_after_t1 = learner.adapted_ptm.adaptmlp.weight.detach().clone()
        self.assertFalse(torch.equal(adapter_after_t1, adapter_after_t0))
        self.assertTrue(torch.equal(learner.adapted_ptm.base.weight, base_weight_init))
        self.assertTrue(torch.equal(learner.classifier.weight[:2], w_old_snapshot))
        self.assertTrue(all(not p.requires_grad for p in learner.adapted_ptm.parameters()))
        self.assertEqual(learner.classifier.weight.shape, (4, 16))

        # Stage 2 (b >= T=2): frozen permanently
        loader2 = self._make_loader(start_cls=4, num_classes=2)
        learner.execute_task(2, loader2, loader2, [4, 5])
        self.assertTrue(torch.equal(learner.adapted_ptm.adaptmlp.weight, adapter_after_t1))
        self.assertTrue(torch.equal(learner.classifier.weight[:2], w_old_snapshot))
        self.assertEqual(learner.classifier.weight.shape, (6, 16))


if __name__ == "__main__":
    unittest.main()
