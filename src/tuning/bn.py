import torch.nn as nn
import logging
from .base import TuningStrategy

class BNTuningStrategy(TuningStrategy):
    """
    Chiến lược Batch Normalization (BN) Tuning.
    Chỉ cập nhật lại running_mean và running_var qua Forward Pass, không dùng Backprop.
    """
    @property
    def requires_training(self) -> bool:
        return True # Vẫn cần data loader để chạy qua mô hình

    @property
    def is_forward_only(self) -> bool:
        return True # Flag báo cho Learner biết không dùng Backprop

    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        # 1. Đóng băng toàn bộ tham số
        for param in backbone.parameters():
            param.requires_grad = False
            
        # 2. Reset thống kê của BN và đặt momentum = None để tính trung bình cộng dồn
        for m in backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
                m.momentum = None 
                
        logging.info("BN Tuning Strategy: Freeze all parameters, reset BN stats (momentum=None).")
        return backbone