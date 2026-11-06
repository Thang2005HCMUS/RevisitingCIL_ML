import torch.nn as nn
import logging
from .base import TuningStrategy

class AdapterStrategy(TuningStrategy):
    """
    Chiến lược Adapter/AdaptFormer.
   [cite: 23]
    """
    @property
    def requires_training(self) -> bool:
        return True

    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        # Trong Adapter, PTM ban đầu thường đã được load kèm cấu trúc Adapter từ hàm get_convnet.
        # Ở đây ta giả định các tham số Adapter đã có tên nhận diện đặc trưng, ví dụ 'adaptmlp' hoặc không có trong bộ pre-trained.
        # Dựa trên logic của tác giả trong aper_adapter.py, ta lấy các tham số requires_grad = True[cite: 3]
        
        # (Lưu ý: Mã gốc xử lý việc gán requires_grad lúc load weight ở hàm vit_base_patch16_224_adapter[cite: 3], 
        # nên ở đây ta chỉ cần giữ nguyên trạng thái hoặc kiểm tra lại).
        
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        logging.info(f"Adapter Strategy: Trainable params set by backbone loader: {trainable_params:,}")
        
        return backbone