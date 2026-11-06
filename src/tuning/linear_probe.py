import torch.nn as nn
from .base import TuningStrategy

class FreezeStrategy(TuningStrategy):
    """
    Chiến lược cho SimpleCIL: Đóng băng toàn bộ mô hình và không huấn luyện.
   [cite: 23]
    """
    @property
    def requires_training(self) -> bool:
        return False

    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        # Đóng băng toàn bộ tham số của PTM
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        return backbone