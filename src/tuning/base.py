from abc import ABC, abstractmethod
import torch.nn as nn

class TuningStrategy(ABC):
    """
    Lớp trừu tượng định nghĩa giao diện chung cho mọi kỹ thuật tinh chỉnh (PET).
   [cite: 23]
    """
    
    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Trả về True nếu chiến lược này cần huấn luyện ở Task 1, False nếu bỏ qua (như SimpleCIL)."""
        pass

    @abstractmethod
    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        """
        Nhận vào mô hình PTM gốc, đóng băng các tham số cần thiết 
        và mở khóa (unfreeze) hoặc cấy thêm các tham số mới.
        """
        pass