import math
import os
import sys
import copy
import logging
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Đã bỏ comment các Import thực tế
from data.task_manager import ContinualTaskManager
from tuning import FreezeStrategy, get_tuning_strategy
from models.backbones import get_pretrained_model
from models.merged_encoder import DualBranchEncoder, VALID_MERGE_MODES
from models.classifiers import CosinePrototypeClassifier
from framework.evaluator import evaluate
from utils.metrics import count_parameters


def _is_wandb_enabled(args: dict) -> bool:
    """Return True when W&B tracking is enabled via CLI or config."""
    return bool(args.get("use_wandb") or args.get("wandb"))


_MERGE_MODE_METHOD_NAMES = {
    "ptm_only": "simplecil_ptm",
    "adapted_only": "simplecil_adapted",
    "dual_ptm": "simplecil_21k+1k",
}


def _resolve_method_name(args: dict) -> str:
    """Resolve canonical Table 1 or Experiment 2 ablation method name."""
    merge_mode = str(args.get("merge_mode") or "both").lower()
    if merge_mode in _MERGE_MODE_METHOD_NAMES:
        method_name = _MERGE_MODE_METHOD_NAMES[merge_mode]
    else:
        model_name = str(args.get("model_name", "unknown"))
        if model_name.lower() == "aper_vpt":
            vpt_type = str(args.get("vpt_type", "shallow")).lower()
            method_name = f"aper_vpt-{vpt_type}"
        else:
            method_name = model_name

    adapt_stages = args.get("adapt_stages")
    if adapt_stages is not None:
        method_name = f"{method_name}_adapt-{int(adapt_stages)}"
    return method_name


def _resolve_wandb_tags(args: dict) -> list:
    """Build W&B run tags including automatic `backbone:<backbone_type>`."""
    tags = []
    backbone_type = args.get("backbone_type") or args.get("convnet_type")
    if backbone_type:
        tags.append(f"backbone:{backbone_type}")

    extra_tags = args.get("wandb_tags")
    if isinstance(extra_tags, str):
        candidates = [t.strip() for t in extra_tags.replace(",", " ").split() if t.strip()]
    elif isinstance(extra_tags, (list, tuple, set)):
        candidates = [str(t).strip() for t in extra_tags if str(t).strip()]
    else:
        candidates = []

    for tag in candidates:
        if tag not in tags:
            tags.append(tag)
    return tags


def _init_wandb(args: dict, learner):
    """Initialize a W&B run when tracking is enabled and return the run object."""
    if not _is_wandb_enabled(args):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B tracking is enabled, but `wandb` is not installed. "
            "Please install it with `pip install wandb`."
        ) from exc

    method_name = _resolve_method_name(args)
    dataset = args.get("dataset", "unknown")
    init_cls = args.get("init_cls", 0)
    increment = args.get("increment", 10)
    seed = args.get("seed", 1)
    backbone_type = args.get("backbone_type") or args.get("convnet_type")
    tags = _resolve_wandb_tags(args)

    total_params = count_parameters(learner.adapted_ptm, trainable=False)
    trainable_params = count_parameters(learner.adapted_ptm, trainable=True)

    entity = args.get("wandb_entity") or "nnm-hcmus"
    project = args.get("wandb_project") or "Group7-ML-Lab"
    group = args.get("wandb_group") or f"{dataset}_init{init_cls}_inc{increment}"
    name = args.get("wandb_name") or f"{method_name}_{dataset}_B{init_cls}_Inc{increment}_seed{seed}"

    config = {}
    for key, value in args.items():
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

    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        name=name,
        tags=tags,
        config=config,
    )

    summary_target = getattr(wandb, "run", None) or run
    if summary_target is not None and getattr(summary_target, "summary", None) is not None:
        summary_target.summary["total_params"] = total_params
        summary_target.summary["trainable_params"] = trainable_params

    learner._wandb = wandb
    learner.wandb_run = run
    return run

def train(args):
    seed_list = copy.deepcopy(args.get("seed", [1]))
    device = copy.deepcopy(args.get("device", ["0"]))
    for seed in seed_list:
        run_args = copy.deepcopy(args)
        run_args["seed"] = seed
        run_args["device"] = copy.deepcopy(device)
        _train(run_args)


def _train(args):
    _set_random(args["seed"])
    _set_device(args)
    _setup_logging(args)

    wandb_run = None
    learner = None
    try:
        logging.info("Khởi tạo Data Manager...")
        data_manager = ContinualTaskManager(args)

        logging.info("Khởi tạo APER Learner...")
        learner = APERLearner(args)
        wandb_run = _init_wandb(args, learner)

        top1_curve, top5_curve = [], []
        for task_id in range(data_manager.nb_tasks):
            logging.info(f"========== Bắt đầu Task {task_id} ==========")
            nb_old = data_manager.known_classes
            train_loader, test_loader, proto_loader = data_manager.get_task_loaders(task_id)
            class_ids = data_manager.get_class_ids(task_id)

            learner.execute_task(task_id, train_loader, proto_loader, class_ids)

            eval_results = learner.evaluate(test_loader, nb_old)
            logging.info(f"Kết quả Task {task_id}: {eval_results}")

            top1 = float(eval_results["top1"])
            top5 = float(eval_results.get("top5", 0.0))
            top1_curve.append(top1)
            top5_curve.append(top5)
            avg_acc_so_far = round(float(np.mean(top1_curve)), 2)

            learner._log_wandb(
                {
                    "task/task_id": task_id,
                    "task/seen_classes": data_manager.known_classes,
                    "eval/top1": top1,
                    "eval/top5": top5,
                    "eval/old_acc": float(eval_results.get("old", 0.0)),
                    "eval/new_acc": float(eval_results.get("new", 0.0)),
                    "eval/avg_acc_so_far": avg_acc_so_far,
                }
            )

        if top1_curve:
            avg_acc = round(float(np.mean(top1_curve)), 2)
            last_acc = round(float(top1_curve[-1]), 2)
            top5_avg_acc = round(float(np.mean(top5_curve)), 2)
            top5_last_acc = round(float(top5_curve[-1]), 2)

            method_name = _resolve_method_name(args)
            logging.info(f"Top-1 curve: {top1_curve}")
            logging.info(f"Top-5 curve: {top5_curve}")
            logging.info(
                f"[{method_name} | {args.get('dataset', 'unknown')} | seed={args.get('seed', 1)}] "
                f"Table 1 Summary => avg_acc: {avg_acc:.2f}, last_acc: {last_acc:.2f}"
            )

            wandb_mod = getattr(learner, "_wandb", None)
            summary_target = (getattr(wandb_mod, "run", None) if wandb_mod is not None else None) or wandb_run
            if summary_target is not None and getattr(summary_target, "summary", None) is not None:
                summary_target.summary["avg_acc"] = avg_acc
                summary_target.summary["last_acc"] = last_acc
                summary_target.summary["table1/avg_acc"] = avg_acc
                summary_target.summary["table1/last_acc"] = last_acc
                summary_target.summary["top5_avg_acc"] = top5_avg_acc
                summary_target.summary["top5_last_acc"] = top5_last_acc
                summary_target.summary["top1_curve"] = top1_curve
                summary_target.summary["top5_curve"] = top5_curve
    finally:
        if wandb_run is not None:
            wandb_mod = getattr(learner, "_wandb", None) if learner is not None else None
            if wandb_mod is not None and hasattr(wandb_mod, "finish"):
                wandb_mod.finish()
            elif hasattr(wandb_run, "finish"):
                wandb_run.finish()


class APERLearner:
    def __init__(self, args):
        self.args = args
        self.device = args["device"][0]
        self.merge_mode = str(args.get("merge_mode") or "both").lower()
        if self.merge_mode not in VALID_MERGE_MODES:
            raise ValueError(
                f"Unsupported merge_mode '{self.merge_mode}'. Expected one of {VALID_MERGE_MODES}."
            )

        # 1. Lấy mô hình tiền huấn luyện PTM
        self.frozen_ptm = get_pretrained_model(args).to(self.device)

        # 2. Lấy chiến lược Tuning và khởi tạo nhánh thứ hai theo merge_mode
        if self.merge_mode == "dual_ptm":
            second_args = copy.deepcopy(args)
            second_args["convnet_type"] = args.get("second_convnet_type", "pretrained_vit_b16_224")
            self.strategy = FreezeStrategy()
            self.adapted_ptm = get_pretrained_model(second_args).to(self.device)
        elif self.merge_mode == "ptm_only":
            self.strategy = FreezeStrategy()
            self.adapted_ptm = self.strategy.prepare_adapted_model(
                copy.deepcopy(self.frozen_ptm)
            ).to(self.device)
        else:
            self.strategy = get_tuning_strategy(args["convnet_type"], args.get("model_name", ""))
            self.adapted_ptm = self.strategy.prepare_adapted_model(
                copy.deepcopy(self.frozen_ptm)
            ).to(self.device)

        # 3. Khởi tạo rỗng cho Encoder và Classifier với số chiều khớp merge_mode
        self.encoder = None
        if self.merge_mode == "ptm_only":
            classifier_in_dim = self.frozen_ptm.out_dim
        elif self.merge_mode == "adapted_only":
            classifier_in_dim = self.adapted_ptm.out_dim
        else:
            classifier_in_dim = self.frozen_ptm.out_dim + self.adapted_ptm.out_dim
        self.classifier = CosinePrototypeClassifier(in_features=classifier_in_dim).to(self.device)
        self._wandb = None
        self.wandb_run = None

    def execute_task(self, task_id: int, train_loader: DataLoader, proto_loader: DataLoader, class_ids: list):
        adapt_stages_raw = self.args.get("adapt_stages", 1)
        adapt_stages = 1 if adapt_stages_raw is None else int(adapt_stages_raw)
        if adapt_stages < 0:
            raise ValueError(f"adapt_stages must be >= 0, got {adapt_stages}")

        nb_old = self.classifier.weight.shape[0]
        if task_id < adapt_stages:
            if task_id > 0:
                self.adapted_ptm = self.strategy.prepare_adapted_model(self.adapted_ptm).to(self.device)
            self._adapt_model(train_loader, nb_old=nb_old, nb_new=len(class_ids), task_id=task_id)

            # Đóng băng nhánh vừa huấn luyện và kết hợp nhánh theo merge_mode
            self.adapted_ptm.eval()
            for param in self.adapted_ptm.parameters():
                param.requires_grad = False

            self.encoder = DualBranchEncoder(self.frozen_ptm, self.adapted_ptm, merge_mode=self.merge_mode)
        elif self.encoder is None:
            self.adapted_ptm.eval()
            for param in self.adapted_ptm.parameters():
                param.requires_grad = False
            self.encoder = DualBranchEncoder(self.frozen_ptm, self.adapted_ptm, merge_mode=self.merge_mode)

        # Giai đoạn 3: Tính toán Prototypes
        self._update_prototypes(proto_loader, class_ids)
        
    def _log_wandb(self, metrics: dict):
        """Log a metrics dictionary to W&B when an active session is attached."""
        if self._wandb is not None and getattr(self._wandb, "run", None) is not None:
            self._wandb.log(metrics)
        elif self.wandb_run is not None and hasattr(self.wandb_run, "log"):
            self.wandb_run.log(metrics)

    def evaluate(self, test_loader, known_classes):
        """Đánh giá mô hình trên tập test."""
        return evaluate(
            self.encoder,
            self.classifier,
            test_loader,
            known_classes,
            self.device,
            increment=self.args.get("increment", 10),
        )

    # ... Giữ nguyên các hàm _adapt_model, _update_prototypes, _set_device, v.v. đã viết ở lần trước ...

    def _adapt_model(self, train_loader, nb_old: int = 0, nb_new: int = None, task_id: int = 0):
        """
        Huấn luyện tham số của adapted_ptm dựa trên chiến lược (Strategy) đã quy định.
        Khi `nb_old > 0` (chặng b >= 1), giữ đóng băng trọng số các lớp cũ W_old và
        chỉ tối ưu tham số PET cùng trọng số lớp mới W_new.
        """
        if not self.strategy.requires_training:
            logging.info("Chiến lược hiện tại không yêu cầu huấn luyện (SimpleCIL). Bỏ qua bước Adapt.")
            return
        if getattr(self.strategy, 'is_forward_only', False):
            logging.info("Bắt đầu tinh chỉnh bằng Forward Pass (BN Tuning)...")
            self.adapted_ptm.train() # Bắt buộc là train() để BN cập nhật running stats
            self.adapted_ptm.to(self.device)
            
            with torch.no_grad(): # Tắt tính toán đạo hàm
                for _, inputs, _ in train_loader:
                    inputs = inputs.to(self.device)
                    _ = self.adapted_ptm(inputs)
                    
            logging.info("Hoàn tất cập nhật Batch Normalization (Không dùng Backprop).")
            return
        epochs = self.args.get("tuned_epoch", 20)
        logging.info(f"Bắt đầu tinh chỉnh (Adaptation) Task {task_id} trong {epochs} epochs...")
        
        self.adapted_ptm.train()
        self.adapted_ptm.to(self.device)

        # 1. Cấu hình Optimizer chỉ cho các tham số được phép huấn luyện (requires_grad = True)
        lr = self.args.get("init_lr", 0.01)
        weight_decay = self.args.get("weight_decay", 0.0005)
        trainable_params = [p for p in self.adapted_ptm.parameters() if p.requires_grad]

        if self.args.get("optimizer", "sgd").lower() == "adam":
            optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(trainable_params, momentum=0.9, lr=lr, weight_decay=weight_decay)

        # 2. Cấu hình Scheduler
        min_lr = self.args.get("min_lr", 1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        # 3. Tạo Classifier tạm thời (chặng 0) hoặc ghép W_old (đóng băng) + W_new (huấn luyện) ở chặng b >= 1
        if nb_new is None:
            nb_new = self.args.get("init_cls", 50)
        feat_dim = self.adapted_ptm.out_dim
        if nb_old > 0:
            w_old = self.classifier.weight[:nb_old, :feat_dim].detach().clone().to(self.device)
            w_old.requires_grad_(False)
            w_new = torch.nn.Parameter(torch.empty(nb_new, feat_dim, device=self.device))
            torch.nn.init.kaiming_uniform_(w_new, a=math.sqrt(5))
            temp_classifier = None
            temp_opt = optim.SGD([w_new], lr=lr)
        else:
            w_old, w_new = None, None
            temp_classifier = torch.nn.Linear(feat_dim, nb_new).to(self.device)
            temp_opt = optim.SGD(temp_classifier.parameters(), lr=lr)

        # 4. Vòng lặp huấn luyện Cross Entropy
        for epoch in range(epochs):
            self.adapted_ptm.train()
            if temp_classifier is not None:
                temp_classifier.train()
            losses, correct, total = 0.0, 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Trích xuất đặc trưng
                features = self.adapted_ptm(inputs)
                if isinstance(features, dict):
                    features = features['features']

                # Tính Cross Entropy Loss
                if nb_old > 0:
                    combined_weight = torch.cat([w_old, w_new], dim=0)
                    logits = 16.0 * F.linear(
                        F.normalize(features, p=2, dim=1),
                        F.normalize(combined_weight, p=2, dim=1),
                    )
                else:
                    logits = temp_classifier(features)
                loss = F.cross_entropy(logits, targets)

                # Cập nhật Gradient
                optimizer.zero_grad()
                temp_opt.zero_grad()
                loss.backward()
                optimizer.step()
                temp_opt.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += len(targets)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            avg_loss = losses / len(train_loader) if len(train_loader) > 0 else 0.0
            train_acc = (correct * 100.0) / total if total > 0 else 0.0

            self._log_wandb(
                {
                    "adapt/epoch": epoch + 1,
                    "adapt/loss": avg_loss,
                    "adapt/train_acc": train_acc,
                    "adapt/lr": current_lr,
                }
            )

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(f"Task {task_id}, Epoch {epoch + 1}/{epochs} => Loss {avg_loss:.3f}, Train Acc {train_acc:.2f}%")

        logging.info("Hoàn tất quá trình Adaptation.")

    def _update_prototypes(self, proto_loader, class_ids):
        """
        Trích xuất trung bình đặc trưng (Prototype) của task hiện tại để thay thế classifier.
        Thay thế cho hàm `replace_fc` trong code gốc.
       [cite: 1, 2, 3]
        """
        logging.info(f"Đang trích xuất Prototype (tâm cụm) cho {len(class_ids)} lớp mới...")
        self.encoder.eval()
        self.encoder.to(self.device)

        features_list, labels_list = [], []

        # 1. Trích xuất toàn bộ embedding của dữ liệu thuộc Task hiện tại
        with torch.no_grad():
            for _, inputs, targets in proto_loader:
                inputs = inputs.to(self.device)
                
                # Đi qua nhánh kép [phi*(x), phi(x)][cite: 6]
                features = self.encoder(inputs) 
                
                features_list.append(features.cpu())
                labels_list.append(targets)

        features_list = torch.cat(features_list, dim=0)
        labels_list = torch.cat(labels_list, dim=0)

        new_prototypes = []

        # 2. Tính trung bình cộng (Mean) của các feature theo từng class
        for class_idx in class_ids:
            idxes = (labels_list == class_idx).nonzero(as_tuple=True)[0]
            if len(idxes) == 0:
                logging.warning(f"Không tìm thấy mẫu nào cho lớp {class_idx}!")
                continue
                
            class_features = features_list[idxes]
            prototype = class_features.mean(dim=0)
            new_prototypes.append(prototype)

        new_prototypes = torch.stack(new_prototypes, dim=0)

        # 3. Gắn các Prototype này vào bộ phân loại Cosine Classifier
        self.classifier.register_prototypes(new_prototypes)
        logging.info("Đã cập nhật Prototypes vào Classifier thành công.")

# ================= Các hàm phụ trợ (Utility) =================
def _set_device(args):
    device_type = args.get("device", ["0"])
    if not isinstance(device_type, (list, tuple)):
        device_type = [device_type]
    gpus = []
    for device in device_type:
        if isinstance(device, torch.device):
            gpus.append(device)
        elif str(device) == "-1" or str(device).lower() == "cpu":
            gpus.append(torch.device("cpu"))
        else:
            gpus.append(torch.device(f"cuda:{device}"))
    args["device"] = gpus

def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _setup_logging(args):
    init_cls = 0 if args.get("init_cls") == args.get("increment") else args.get("init_cls", 0)
    method_name = _resolve_method_name(args)
    log_dir = f"logs/{method_name}/{args.get('dataset', 'unknown')}/{init_cls}/{args.get('increment', 0)}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logfilename = f"{log_dir}/{args.get('prefix', 'exp')}_{args.get('seed', 1)}_{args.get('convnet_type', 'vit')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )