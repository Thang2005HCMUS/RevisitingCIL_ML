import torch.nn as nn
import logging
from .base import TuningStrategy

class FinetuneStrategy(TuningStrategy):
    """
    Chiến lược Fully Finetune.
    Mở khóa và huấn luyện toàn bộ tham số của mô hình tiền huấn luyện (PTM)
    bằng lan truyền ngược (Backpropagation).
    """
    @property
    def requires_training(self) -> bool:
        return True

    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        # Mở khóa toàn bộ tham số của backbone
        for param in backbone.parameters():
            param.requires_grad = True
            
        # Log thông tin để kiểm tra
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        
        logging.info(f"Finetune Strategy: Mở khóa toàn bộ tham số.")
        logging.info(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")
        
        return backbone