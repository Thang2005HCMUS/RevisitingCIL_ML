import torch
import torch.nn as nn
import torch.nn.functional as F

class CosinePrototypeClassifier(nn.Module):
    """
    Bộ phân loại dựa trên Cosine Similarity và Prototypes.
    Tự động mở rộng khi có các lớp mới được thêm vào.
   
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        
        # Khởi tạo tensor rỗng cho trọng số (weights), sẽ được thêm dần qua các Task
        self.weight = nn.Parameter(torch.Tensor(0, in_features), requires_grad=False)
        
        # Hệ số scale (Sigma) để tối ưu logits cho Softmax (Tương tự CosineLinear trong code gốc)[cite: 3]
        self.sigma = nn.Parameter(torch.Tensor(1))
        self.sigma.data.fill_(1.0)
        
    def register_prototypes(self, new_prototypes: torch.Tensor):
        """
        Nối thêm các prototype (tâm cụm) của các lớp mới vào ma trận trọng số.
        new_prototypes: Tensor có kích thước [num_new_classes, in_features]
       
        """
        device = self.weight.device
        new_prototypes = new_prototypes.to(device)
        
        # Nối prototype cũ và mới theo chiều không gian các lớp (dim=0)
        updated_weight = torch.cat([self.weight.data, new_prototypes], dim=0)
        
        # Cập nhật lại Parameter
        self.weight = nn.Parameter(updated_weight, requires_grad=False)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Tính toán normalized cosine similarity: (w / ||w||) * (f / ||f||).
       [cite: 23]
        """
        # Chuẩn hóa (L2 Normalize) đặc trưng đầu vào và trọng số prototype
        norm_features = F.normalize(features, p=2, dim=1)
        norm_weight = F.normalize(self.weight, p=2, dim=1)
        
        # Tính Cosine Similarity thông qua phép nhân ma trận
        logits = F.linear(norm_features, norm_weight)
        
        # Nhân với hằng số scale sigma[cite: 3]
        if self.sigma is not None:
            logits = self.sigma * logits
            
        return logits