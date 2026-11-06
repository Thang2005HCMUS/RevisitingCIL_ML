import torch.nn as nn
import logging
from .base import TuningStrategy

class PromptStrategy(TuningStrategy):
    """
    Chiến lược Visual Prompt Tuning (VPT).
    Hỗ trợ cả VPT-Shallow và VPT-Deep tùy thuộc vào cấu trúc Prompt_Tokens được cấy vào backbone.
   
    """
    @property
    def requires_training(self) -> bool:
        return True

    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        """
        Đóng băng toàn bộ tham số của mô hình gốc và chỉ mở khóa các Prompt_Tokens.
       [cite: 3]
        """
        # Nếu backbone có sẵn hàm Freeze() (như trong class VPT_ViT gốc của tác giả), ta có thể gọi nó.
        # Tuy nhiên, để đảm bảo tính nhất quán của Strategy Pattern, ta quét qua named_parameters.
        for name, param in backbone.named_parameters():
            # Mở khóa các tham số thuộc Prompt, đóng băng phần còn lại
            if "Prompt_Tokens" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # Kiểm tra và log số lượng tham số huấn luyện[cite: 3]
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        logging.info(f"VPT Strategy: Total params: {total_params:,} | Trainable params: {trainable_params:,}")
        
        return backbone