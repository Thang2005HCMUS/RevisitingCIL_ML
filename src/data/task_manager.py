import logging
import numpy as np
from torch.utils.data import DataLoader

class ContinualTaskManager:
    """
    Quản lý tập dữ liệu (Dataset) và phân chia thành các CIL Tasks (Sessions).
  
    """
    def __init__(self, args):
        self.dataset_name = args.get("dataset", "cifar100").lower()
        self.shuffle = args.get("shuffle", True)
        self.seed = args.get("seed", 1)
        self.init_cls = args.get("init_cls", 50)
        self.increment = args.get("increment", 10)
        self.batch_size = args.get("batch_size", 128)
        self.num_workers = args.get("num_workers", 4)
        
        # Số lớp đã học được
        self.known_classes = 0
        
        # 1. Setup dữ liệu gốc và thứ tự các lớp (class order)
        self._setup_data()
        
        # 2. Tính toán phân bổ số lượng lớp cho từng Task
        self._increments = [self.init_cls]
        while sum(self._increments) + self.increment <= len(self._class_order):
            self._increments.append(self.increment)
            
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self) -> int:
        """Tổng số Tasks trong toàn bộ tiến trình học."""
        return len(self._increments)

    def get_class_ids(self, task_id: int) -> list:
        """Trả về danh sách các Class ID xuất hiện trong Task hiện tại."""
        start = sum(self._increments[:task_id])
        end = start + self._increments[task_id]
        return np.arange(start, end).tolist()
        
    def get_task_loaders(self, task_id: int):
        """
        Cung cấp 3 DataLoaders phục vụ cho vòng lặp học:
        - train_loader: Huấn luyện PTM (Có Augmentation).
        - test_loader: Đánh giá mô hình trên toàn bộ các lớp ĐÃ THẤY (No Augmentation).
        - proto_loader: Trích xuất đặc trưng nguyên mẫu cho lớp mới (No Augmentation).
        """
        task_size = self._increments[task_id]
        start_cls = self.known_classes
        end_cls = self.known_classes + task_size
        
        logging.info(f"Chuẩn bị dữ liệu Task {task_id} (Lớp {start_cls} đến {end_cls - 1})")
        
        new_classes = np.arange(start_cls, end_cls)
        all_seen_classes = np.arange(0, end_cls)
        
        # Giả định _get_dataset sẽ bọc lấy các hàm get_dataset từ `utils/data_manager.py` của code cũ
        train_dataset = self._get_dataset(new_classes, source="train", mode="train")
        test_dataset = self._get_dataset(all_seen_classes, source="test", mode="test")
        proto_dataset = self._get_dataset(new_classes, source="train", mode="test")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        proto_loader = DataLoader(proto_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        # Cập nhật số lớp đã thấy cho lần gọi Task tiếp theo
        self.known_classes = end_cls
        
        return train_loader, test_loader, proto_loader
        pass

    def _setup_data(self):
        self.data_source = get_dataset(self.dataset_name)
        self.data_source.download_data()
        
        # Lấy thứ tự các lớp (Có thể trộn ngẫu nhiên nếu self.shuffle = True)
        order = list(range(len(np.unique(self.data_source.train_targets))))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(order)
        self._class_order = order