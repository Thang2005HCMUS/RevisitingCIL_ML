import torch.nn as nn
import logging
from .base import TuningStrategy

class SSFStrategy(TuningStrategy):
    """
    Chiến lược Scale & Shift (SSF). Chỉ huấn luyện các tham số scale/shift.
   [cite: 23]
    """
    @property
    def requires_training(self) -> bool:
        return True

    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        # Logic freeze của APER-SSF (từ mã nguồn gốc của tác giả)[cite: 3]
        for name, param in backbone.named_parameters():
            # Mở khóa các tham số thuộc SSF, đóng băng phần còn lại
            if "ssf_scale" not in name and "ssf_shift_" not in name: 
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        # Kiểm tra và log số lượng tham số huấn luyện[cite: 3]
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        logging.info(f"SSF Strategy: Total params: {total_params:,} | Trainable params: {trainable_params:,}")
        
        return backbone