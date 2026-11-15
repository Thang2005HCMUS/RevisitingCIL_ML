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
        # Chủ động mở khóa các tham số của adapter
        for name, param in backbone.named_parameters():
            if "adaptmlp" in name or "scale" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        logging.info(f"Adapter Strategy: Trainable params set by backbone loader: {trainable_params:,}")
        
        return backbone