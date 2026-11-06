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

# Lưu ý: Các thư viện này sẽ được implement ở các bước tiếp theo
from data.task_manager import ContinualTaskManager
from tuning import get_tuning_strategy
from models.backbones import get_pretrained_model
from models.merged_encoder import DualBranchEncoder
from models.classifiers import CosinePrototypeClassifier
from framework.evaluator import evaluate

def train(args):
    """
    Hàm khởi chạy vòng lặp huấn luyện qua nhiều seed khác nhau.
    Giữ nguyên logic của tác giả trong trainer.py.
    """
    seed_list = copy.deepcopy(args.get("seed", [1]))
    device = copy.deepcopy(args.get("device", ["0"]))

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    """
    Cài đặt môi trường và chạy vòng lặp CIL Tasks.
    """
    _set_random(args["seed"])
    _set_device(args)
    _setup_logging(args)
    
    logging.info("Khởi tạo Data Manager...")
    data_manager = ContinualTaskManager(args)
    
    logging.info("Khởi tạo APER Learner...")
    learner = APERLearner(args)
    
    # Vòng lặp Class-Incremental Learning (CIL)
    for task_id in range(data_manager.nb_tasks):
        logging.info(f"========== Bắt đầu Task {task_id} ==========")
        
        train_loader, test_loader, proto_loader = data_manager.get_task_loaders(task_id)
        class_ids = data_manager.get_class_ids(task_id)
    
        learner.execute_task(task_id, train_loader, proto_loader, class_ids)
        
        # Đánh giá sau khi học xong task
        eval_results = learner.evaluate(test_loader, data_manager.known_classes)
        logging.info(f"Kết quả Task {task_id}: {eval_results}")

class APERLearner:
    """
    Thiết kế theo Template Method Pattern.
    Điều phối vòng đời: Task 1 (Adapt -> Merge -> Prototypes), Task 2...B (Chỉ Prototypes).
    """
    def __init__(self, args):
        self.args = args
        self.device = args["device"][0]
        
        1. Lấy mô hình tiền huấn luyện PTM
        self.frozen_ptm = get_pretrained_model(args["convnet_type"]).to(self.device)
        self.frozen_ptm.eval()
        for param in self.frozen_ptm.parameters():
            param.requires_grad = False
            
        2. Lấy chiến lược Tuning (Adapter, VPT, SSF, Finetune, hoặc SimpleCIL - Freeze)
        self.strategy = get_tuning_strategy(args["convnet_type"])
        
        self.adapted_ptm = self.strategy.prepare_adapted_model(copy.deepcopy(self.frozen_ptm)).to(self.device)
        
        3. Bộ phân loại Cosine và Hợp nhất đặc trưng
        self.encoder = None
        self.classifier = CosinePrototypeClassifier(in_features=self.frozen_ptm.out_dim * 2).to(self.device)

    def execute_task(self, task_id: int, train_loader: DataLoader, proto_loader: DataLoader, class_ids: list):
        """Luồng thực thi chính cho mỗi Task."""
        if task_id == 0:
            # Giai đoạn 1: Thích ứng mô hình với dữ liệu (Chỉ chạy ở Task đầu tiên)
            self._adapt_model(train_loader)
            
            Giai đoạn 2: Đóng băng nhánh vừa huấn luyện và kết hợp 2 nhánh [phi*, phi]
            self.adapted_ptm.eval()
            for param in self.adapted_ptm.parameters():
                param.requires_grad = False
            self.encoder = DualBranchEncoder(self.frozen_ptm, self.adapted_ptm)
        
        # Giai đoạn 3: Tính toán Prototypes và mở rộng Classifier
        self._update_prototypes(proto_loader, class_ids)

    def _adapt_model(self, train_loader):
        """
        Huấn luyện tham số của adapted_ptm dựa trên chiến lược (Strategy) đã quy định.
       [cite: 1, 2]
        """
        if not self.strategy.requires_training:
            logging.info("Chiến lược hiện tại không yêu cầu huấn luyện (SimpleCIL). Bỏ qua bước Adapt.")
            return

        epochs = self.args.get("tuned_epoch", 20)
        logging.info(f"Bắt đầu tinh chỉnh (Adaptation) trong {epochs} epochs...")
        
        self.adapted_ptm.train()
        self.adapted_ptm.to(self.device)

        # 1. Cấu hình Optimizer chỉ cho các tham số được phép huấn luyện (requires_grad = True)
        lr = self.args.get("init_lr", 0.01)
        weight_decay = self.args.get("weight_decay", 0.0005)
        trainable_params = [p for p in self.adapted_ptm.parameters() if p.requires_grad]

        if self.args.get("optimizer", "sgd").lower() == "adam":
            optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)[cite: 1, 2]
        else:
            optimizer = optim.SGD(trainable_params, momentum=0.9, lr=lr, weight_decay=weight_decay)[cite: 1, 2]

        # 2. Cấu hình Scheduler
        min_lr = self.args.get("min_lr", 1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)[cite: 1, 2]

        # 3. Tạo một Classifier tuyến tính TẠM THỜI để tính Loss trong quá trình huấn luyện Task 1
        temp_classifier = torch.nn.Linear(self.adapted_ptm.out_dim, self.args.get('init_cls', 50)).to(self.device)
        temp_opt = optim.SGD(temp_classifier.parameters(), lr=lr)

        # 4. Vòng lặp huấn luyện Cross Entropy
        for epoch in range(epochs):
            self.adapted_ptm.train()
            temp_classifier.train()
            losses, correct, total = 0.0, 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Trích xuất đặc trưng
                features = self.adapted_ptm(inputs)
                if isinstance(features, dict):
                    features = features['features']

                # Tính Cross Entropy Loss[cite: 2]
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

            scheduler.step()
            train_acc = (correct * 100.0) / total
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(f"Task 0, Epoch {epoch + 1}/{epochs} => Loss {losses/len(train_loader):.3f}, Train Acc {train_acc:.2f}%")

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

    def evaluate(self, test_loader, known_classes):
        """Đánh giá mô hình trên tập test."""
        return evaluate(self.encoder, self.classifier, test_loader, known_classes, self.device)
        pass

# ================= Các hàm phụ trợ (Utility) =================
def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
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
    log_dir = f"logs/{args.get('model_name', 'aper')}/{args.get('dataset', 'unknown')}/{init_cls}/{args.get('increment', 0)}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logfilename = f"{log_dir}/{args.get('prefix', 'exp')}_{args.get('seed', 1)}_{args.get('convnet_type', 'vit')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )