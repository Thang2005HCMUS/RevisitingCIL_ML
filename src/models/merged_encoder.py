import torch
import torch.nn as nn

class DualBranchEncoder(nn.Module):
    """
    Module hợp nhất đặc trưng từ mô hình PTM gốc và mô hình đã thích ứng (Adapted).
    Tương đương với [phi*(x), phi(x)] trong bài báo APER.
   [cite: 6, 23]
    """
    def __init__(self, frozen_ptm: nn.Module, adapted_ptm: nn.Module):
        super().__init__()
        self.frozen_ptm = frozen_ptm
        self.adapted_ptm = adapted_ptm
        
        # Đảm bảo cả hai mô hình đều ở trạng thái eval và không cập nhật gradient
        # vì giai đoạn Adapt đã kết thúc.
        self.frozen_ptm.eval()
        self.adapted_ptm.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất và nối đặc trưng từ cả hai nhánh."""
        with torch.no_grad():
            feat_adapted = self.adapted_ptm(x)
            feat_frozen = self.frozen_ptm(x)
            
            # Xử lý trường hợp mô hình gốc trả về dict (dựa theo code của tác giả)
            if isinstance(feat_adapted, dict):
                feat_adapted = feat_adapted['features']
            if isinstance(feat_frozen, dict):
                feat_frozen = feat_frozen['features']
                
            # Hợp nhất đặc trưng theo chiều channel: [phi*(x), phi(x)]
            merged_features = torch.cat([feat_adapted, feat_frozen], dim=-1)
            
        return merged_features