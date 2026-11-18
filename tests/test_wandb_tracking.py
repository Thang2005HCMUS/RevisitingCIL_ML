import logging
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import main as cli_main
from framework.learner import APERLearner, _init_wandb, _resolve_method_name, _setup_logging, _train


class TestWandbTrackingTask21(unittest.TestCase):
    def test_resolve_method_name_vpt_variants(self):
        self.assertEqual(
            _resolve_method_name({"model_name": "aper_vpt", "vpt_type": "deep"}),
            "aper_vpt-deep",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "aper_vpt", "vpt_type": "shallow"}),
            "aper_vpt-shallow",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "aper_vpt"}),
            "aper_vpt-shallow",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "simplecil"}),
            "simplecil",
        )

    def test_resolve_method_name_exp2_variants(self):
        self.assertEqual(
            _resolve_method_name({"model_name": "simplecil", "merge_mode": "ptm_only"}),
            "simplecil_ptm",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "aper_finetune", "merge_mode": "adapted_only"}),
            "simplecil_adapted",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "simplecil", "merge_mode": "dual_ptm"}),
            "simplecil_21k+1k",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "aper_finetune", "merge_mode": "both"}),
            "aper_finetune",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "aper_adapter", "adapt_stages": 0}),
            "aper_adapter_adapt-0",
        )
        self.assertEqual(
            _resolve_method_name({"model_name": "aper_adapter", "adapt_stages": 3}),
            "aper_adapter_adapt-3",
        )

    def test_init_wandb_exp2_run_naming_and_config(self):
        dummy_learner = types.SimpleNamespace(adapted_ptm=nn.Linear(4, 2), _wandb=None, wandb_run=None)
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_wandb = types.ModuleType("wandb")
        mock_wandb.run = mock_run
        mock_wandb.init = MagicMock(return_value=mock_run)

        orig_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = mock_wandb
        try:
            args = {
                "use_wandb": True,
                "model_name": "aper_adapter",
                "adapt_stages": 0,
                "dataset": "cifar224",
                "init_cls": 10,
                "increment": 10,
                "seed": 1993,
                "device": [torch.device("cpu")],
            }
            _init_wandb(args, dummy_learner)
        finally:
            if orig_wandb is not None:
                sys.modules["wandb"] = orig_wandb
            else:
                sys.modules.pop("wandb", None)

        kwargs = mock_wandb.init.call_args.kwargs
        self.assertEqual(kwargs["group"], "cifar224_init10_inc10")
        self.assertEqual(kwargs["name"], "aper_adapter_adapt-0_cifar224_B10_Inc10_seed1993")
        self.assertEqual(kwargs["config"]["method_name"], "aper_adapter_adapt-0")
        self.assertEqual(kwargs["config"]["adapt_stages"], 0)

    def test_init_wandb_disabled_returns_none(self):
        dummy_learner = types.SimpleNamespace(adapted_ptm=nn.Linear(4, 2))
        self.assertIsNone(_init_wandb({"model_name": "simplecil"}, dummy_learner))

    def test_init_wandb_enabled_records_params_and_defaults(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        for p in model[0].parameters():
            p.requires_grad = False
        # model[0]: 4*3 + 3 = 15 (frozen); model[1]: 3*2 + 2 = 8 (trainable); total = 23
        dummy_learner = types.SimpleNamespace(adapted_ptm=model, _wandb=None, wandb_run=None)

        mock_run = MagicMock()
        mock_run.summary = {}
        mock_wandb = types.ModuleType("wandb")
        mock_wandb.run = mock_run
        mock_wandb.init = MagicMock(return_value=mock_run)

        orig_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = mock_wandb
        try:
            args = {
                "use_wandb": True,
                "model_name": "aper_vpt",
                "vpt_type": "deep",
                "dataset": "cifar224",
                "init_cls": 5,
                "increment": 5,
                "seed": 1993,
                "device": [torch.device("cpu")],
            }
            run = _init_wandb(args, dummy_learner)
        finally:
            if orig_wandb is not None:
                sys.modules["wandb"] = orig_wandb
            else:
                sys.modules.pop("wandb", None)

        self.assertIs(run, mock_run)
        mock_wandb.init.assert_called_once()
        kwargs = mock_wandb.init.call_args.kwargs
        self.assertEqual(kwargs["entity"], "nnm-hcmus")
        self.assertEqual(kwargs["project"], "Group7-ML-Lab")
        self.assertEqual(kwargs["group"], "cifar224_init5_inc5")
        self.assertEqual(kwargs["name"], "aper_vpt-deep_cifar224_B5_Inc5_seed1993")
        self.assertEqual(kwargs["config"]["method_name"], "aper_vpt-deep")
        self.assertEqual(kwargs["config"]["total_params"], 23)
        self.assertEqual(kwargs["config"]["trainable_params"], 8)
        self.assertEqual(kwargs["config"]["device"], ["cpu"])
        self.assertEqual(mock_run.summary["total_params"], 23)
        self.assertEqual(mock_run.summary["trainable_params"], 8)


class TestWandbTrackingTask22(unittest.TestCase):
    def test_evaluate_passes_increment(self):
        learner = APERLearner.__new__(APERLearner)
        learner.args = {"increment": 5}
        learner.encoder = object()
        learner.classifier = object()
        learner.device = torch.device("cpu")

        with patch("framework.learner.evaluate", return_value={"top1": 90.0}) as mock_eval:
            result = learner.evaluate("loader", 15)

        self.assertEqual(result, {"top1": 90.0})
        mock_eval.assert_called_once_with(
            learner.encoder,
            learner.classifier,
            "loader",
            15,
            learner.device,
            increment=5,
        )

    def test_adapt_model_logs_epoch_metrics_to_wandb(self):
        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(4, 4)
                self.out_dim = 4

            def forward(self, x):
                return self.net(x)

        learner = APERLearner.__new__(APERLearner)
        learner.args = {
            "tuned_epoch": 2,
            "init_lr": 0.01,
            "weight_decay": 0.0,
            "optimizer": "sgd",
            "min_lr": 1e-5,
            "init_cls": 2,
        }
        learner.device = torch.device("cpu")
        learner.strategy = types.SimpleNamespace(requires_training=True, is_forward_only=False)
        learner.adapted_ptm = TinyBackbone()

        mock_run = MagicMock()
        mock_wandb = types.SimpleNamespace(run=mock_run, log=MagicMock())
        learner._wandb = mock_wandb
        learner.wandb_run = mock_run

        inputs = torch.randn(6, 4)
        targets = torch.tensor([0, 1, 0, 1, 0, 1])
        indices = torch.arange(6)
        train_loader = DataLoader(TensorDataset(indices, inputs, targets), batch_size=3)

        learner._adapt_model(train_loader)

        self.assertEqual(mock_wandb.log.call_count, 2)
        for epoch_idx, call in enumerate(mock_wandb.log.call_args_list, start=1):
            payload = call.args[0]
            self.assertEqual(
                set(payload.keys()),
                {"adapt/epoch", "adapt/loss", "adapt/train_acc", "adapt/lr"},
            )
            self.assertEqual(payload["adapt/epoch"], epoch_idx)
            self.assertIsInstance(payload["adapt/loss"], float)
            self.assertIsInstance(payload["adapt/train_acc"], float)
            self.assertIsInstance(payload["adapt/lr"], float)


class TestWandbTrackingTask23(unittest.TestCase):
    def _build_fake_data_manager(self):
        class FakeDataManager:
            def __init__(self, args):
                self.nb_tasks = 2
                self.known_classes = 0

            def get_task_loaders(self, task_id):
                self.known_classes += 5
                return "train_loader", "test_loader", "proto_loader"

            def get_class_ids(self, task_id):
                start = task_id * 5
                return list(range(start, start + 5))

        return FakeDataManager

    def _build_fake_learner_cls(self, eval_sequence, raise_on_task=None):
        class FakeLearner:
            def __init__(self, args):
                self.args = args
                self.device = args["device"][0]
                self.adapted_ptm = nn.Linear(4, 2)
                self._wandb = None
                self.wandb_run = None

            def execute_task(self, task_id, train_loader, proto_loader, class_ids):
                if raise_on_task is not None and task_id == raise_on_task:
                    raise RuntimeError("boom")

            def evaluate(self, test_loader, nb_old):
                return eval_sequence[0 if nb_old == 0 else 1]

            def _log_wandb(self, metrics):
                if self._wandb is not None and getattr(self._wandb, "run", None) is not None:
                    self._wandb.log(metrics)

        return FakeLearner

    def test_train_end_to_end_dry_run_with_mocked_wandb(self):
        eval_sequence = [
            {"top1": 90.0, "top5": 99.0, "old": 0.0, "new": 90.0},
            {"top1": 80.0, "top5": 95.0, "old": 82.0, "new": 78.0},
        ]
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_wandb = types.ModuleType("wandb")
        mock_wandb.run = mock_run
        mock_wandb.init = MagicMock(return_value=mock_run)
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()

        args = {
            "use_wandb": True,
            "model_name": "aper_vpt",
            "vpt_type": "shallow",
            "dataset": "cifar224",
            "init_cls": 5,
            "increment": 5,
            "seed": 1993,
            "device": [-1],
            "prefix": "test",
            "convnet_type": "vit",
        }

        orig_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = mock_wandb
        try:
            with patch("framework.learner.ContinualTaskManager", self._build_fake_data_manager()), \
                 patch("framework.learner.APERLearner", self._build_fake_learner_cls(eval_sequence)), \
                 patch("framework.learner._setup_logging"):
                _train(args)
        finally:
            if orig_wandb is not None:
                sys.modules["wandb"] = orig_wandb
            else:
                sys.modules.pop("wandb", None)

        mock_wandb.init.assert_called_once()
        self.assertEqual(mock_wandb.log.call_count, 2)

        task0_log = mock_wandb.log.call_args_list[0].args[0]
        self.assertEqual(
            set(task0_log.keys()),
            {
                "task/task_id",
                "task/seen_classes",
                "eval/top1",
                "eval/top5",
                "eval/old_acc",
                "eval/new_acc",
                "eval/avg_acc_so_far",
            },
        )
        self.assertEqual(task0_log["task/task_id"], 0)
        self.assertEqual(task0_log["task/seen_classes"], 5)
        self.assertEqual(task0_log["eval/avg_acc_so_far"], 90.0)

        task1_log = mock_wandb.log.call_args_list[1].args[0]
        self.assertEqual(task1_log["task/task_id"], 1)
        self.assertEqual(task1_log["task/seen_classes"], 10)
        self.assertEqual(task1_log["eval/avg_acc_so_far"], 85.0)

        self.assertEqual(mock_run.summary["avg_acc"], 85.0)
        self.assertEqual(mock_run.summary["last_acc"], 80.0)
        self.assertEqual(mock_run.summary["table1/avg_acc"], 85.0)
        self.assertEqual(mock_run.summary["table1/last_acc"], 80.0)
        self.assertEqual(mock_run.summary["top5_avg_acc"], 97.0)
        self.assertEqual(mock_run.summary["top5_last_acc"], 95.0)
        self.assertEqual(mock_run.summary["top1_curve"], [90.0, 80.0])
        self.assertEqual(mock_run.summary["top5_curve"], [99.0, 95.0])
        mock_wandb.finish.assert_called_once()

    def test_train_calls_wandb_finish_on_exception(self):
        eval_sequence = [{"top1": 90.0, "top5": 99.0, "old": 0.0, "new": 90.0}]
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_wandb = types.ModuleType("wandb")
        mock_wandb.run = mock_run
        mock_wandb.init = MagicMock(return_value=mock_run)
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()

        args = {
            "use_wandb": True,
            "model_name": "simplecil",
            "dataset": "cifar224",
            "init_cls": 5,
            "increment": 5,
            "seed": 1,
            "device": ["cpu"],
        }

        orig_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = mock_wandb
        try:
            with patch("framework.learner.ContinualTaskManager", self._build_fake_data_manager()), \
                 patch("framework.learner.APERLearner", self._build_fake_learner_cls(eval_sequence, raise_on_task=0)), \
                 patch("framework.learner._setup_logging"):
                with self.assertRaises(RuntimeError):
                    _train(args)
        finally:
            if orig_wandb is not None:
                sys.modules["wandb"] = orig_wandb
            else:
                sys.modules.pop("wandb", None)

        mock_wandb.finish.assert_called_once()

    def test_train_without_wandb_never_imports_wandb(self):
        eval_sequence = [
            {"top1": 90.0, "top5": 99.0, "old": 0.0, "new": 90.0},
            {"top1": 80.0, "top5": 95.0, "old": 82.0, "new": 78.0},
        ]
        args = {
            "model_name": "simplecil",
            "dataset": "cifar224",
            "init_cls": 5,
            "increment": 5,
            "seed": 1,
            "device": [-1],
        }
        with patch("framework.learner.ContinualTaskManager", self._build_fake_data_manager()), \
             patch("framework.learner.APERLearner", self._build_fake_learner_cls(eval_sequence)), \
             patch("framework.learner._setup_logging"):
            _train(args)


class TestMainCliOverrides(unittest.TestCase):
    def test_main_cli_overrides_preserve_zero_adapt_stages_and_merge_mode(self):
        json_param = {
            "model_name": "aper_adapter",
            "dataset": "cifar224",
            "adapt_stages": 1,
            "merge_mode": "both",
        }
        argv = [
            "main.py",
            "--config",
            "dummy.json",
            "--adapt_stages",
            "0",
            "--merge_mode",
            "ptm_only",
            "--wandb",
        ]
        with patch.object(sys, "argv", argv), \
             patch.object(cli_main, "load_json", return_value=json_param), \
             patch.object(cli_main, "train") as mock_train:
            cli_main.main()

        mock_train.assert_called_once()
        passed = mock_train.call_args.args[0]
        self.assertEqual(passed["adapt_stages"], 0)
        self.assertEqual(passed["merge_mode"], "ptm_only")
        self.assertTrue(passed["use_wandb"])

    def test_main_without_exp2_flags_does_not_inject_none(self):
        json_param = {
            "model_name": "simplecil",
            "dataset": "cifar224",
        }
        argv = ["main.py", "--config", "dummy.json"]
        with patch.object(sys, "argv", argv), \
             patch.object(cli_main, "load_json", return_value=json_param), \
             patch.object(cli_main, "train") as mock_train:
            cli_main.main()

        passed = mock_train.call_args.args[0]
        self.assertNotIn("adapt_stages", passed)
        self.assertNotIn("merge_mode", passed)


class TestSetupLoggingIsolation(unittest.TestCase):
    def _capture_log_path(self, args: dict) -> str:
        root_logger = logging.getLogger()
        orig_handlers = list(root_logger.handlers)
        orig_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                root_logger.handlers.clear()
                _setup_logging(args)
                file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
                self.assertEqual(len(file_handlers), 1)
                return os.path.relpath(file_handlers[0].baseFilename, tmpdir).replace("\\", "/")
            finally:
                for h in list(root_logger.handlers):
                    if h not in orig_handlers:
                        h.close()
                root_logger.handlers[:] = orig_handlers
                os.chdir(orig_cwd)

    def test_exp2b_adapt_stages_write_to_separate_log_files(self):
        base = {
            "model_name": "aper_adapter",
            "dataset": "cifar224",
            "init_cls": 10,
            "increment": 10,
            "prefix": "exp",
            "seed": 1993,
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
        }
        path_0 = self._capture_log_path({**base, "adapt_stages": 0})
        path_1 = self._capture_log_path({**base, "adapt_stages": 1})
        self.assertNotEqual(path_0, path_1)
        self.assertTrue(path_0.startswith("logs/aper_adapter_adapt-0/"))
        self.assertTrue(path_1.startswith("logs/aper_adapter_adapt-1/"))

    def test_exp2a_simplecil_ptm_and_21k_1k_write_to_separate_log_files(self):
        base = {
            "model_name": "simplecil",
            "dataset": "imageneta",
            "init_cls": 100,
            "increment": 5,
            "prefix": "exp",
            "seed": 1993,
            "convnet_type": "pretrained_vit_b16_224_in21k",
        }
        path_ptm = self._capture_log_path({**base, "merge_mode": "ptm_only"})
        path_dual = self._capture_log_path({**base, "merge_mode": "dual_ptm"})
        self.assertNotEqual(path_ptm, path_dual)
        self.assertTrue(path_ptm.startswith("logs/simplecil_ptm/"))
        self.assertTrue(path_dual.startswith("logs/simplecil_21k+1k/"))

    def test_exp2a_simplecil_adapted_and_aper_finetune_write_to_separate_log_files(self):
        base = {
            "model_name": "aper_finetune",
            "dataset": "imageneta",
            "init_cls": 100,
            "increment": 5,
            "prefix": "exp",
            "seed": 1993,
            "convnet_type": "pretrained_vit_b16_224_in21k",
        }
        path_adapted = self._capture_log_path({**base, "merge_mode": "adapted_only"})
        path_both = self._capture_log_path({**base, "merge_mode": "both"})
        self.assertNotEqual(path_adapted, path_both)
        self.assertTrue(path_adapted.startswith("logs/simplecil_adapted/"))
        self.assertTrue(path_both.startswith("logs/aper_finetune/"))


if __name__ == "__main__":
    unittest.main()
