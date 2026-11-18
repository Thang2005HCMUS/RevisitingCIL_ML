import torch
import torch.nn as nn

VALID_MERGE_MODES = ("both", "adapted_only", "ptm_only", "dual_ptm")


def _get_module_out_dim(module: nn.Module) -> int:
    """Resolve feature dimension exposed by a backbone module."""
    for attr in ("out_dim", "num_features", "embed_dim"):
        val = getattr(module, attr, None)
        if val is not None:
            return int(val)
    raise ValueError(f"Cannot determine output dimension for module {type(module).__name__}.")


def _extract_features(branch: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run forward pass and unwrap dict outputs returned by some backbones."""
    feat = branch(x)
    if isinstance(feat, dict):
        feat = feat["features"]
    return feat


class DualBranchEncoder(nn.Module):
    """
    Module hợp nhất đặc trưng từ mô hình PTM gốc và mô hình đã thích ứng (Adapted).
    Hỗ trợ 4 chế độ ghép (`merge_mode`): "both", "adapted_only", "ptm_only", "dual_ptm".
    """

    def __init__(self, frozen_ptm: nn.Module, adapted_ptm: nn.Module, merge_mode: str = "both"):
        super().__init__()
        mode = str(merge_mode or "both").lower()
        if mode not in VALID_MERGE_MODES:
            raise ValueError(f"Unsupported merge_mode '{merge_mode}'. Expected one of {VALID_MERGE_MODES}.")

        if mode in ("both", "dual_ptm") and (frozen_ptm is None or adapted_ptm is None):
            raise ValueError(f"merge_mode '{mode}' requires both frozen_ptm and adapted_ptm.")
        if mode == "ptm_only" and frozen_ptm is None:
            raise ValueError("merge_mode 'ptm_only' requires frozen_ptm.")
        if mode == "adapted_only" and adapted_ptm is None:
            raise ValueError("merge_mode 'adapted_only' requires adapted_ptm.")

        self.merge_mode = mode
        self.frozen_ptm = frozen_ptm
        self.adapted_ptm = adapted_ptm

        if self.merge_mode == "ptm_only":
            self.out_dim = _get_module_out_dim(self.frozen_ptm)
        elif self.merge_mode == "adapted_only":
            self.out_dim = _get_module_out_dim(self.adapted_ptm)
        else:
            self.out_dim = _get_module_out_dim(self.adapted_ptm) + _get_module_out_dim(self.frozen_ptm)

        # Đảm bảo các nhánh ở trạng thái eval và đóng băng gradient.
        if self.frozen_ptm is not None:
            self.frozen_ptm.eval()
        if self.adapted_ptm is not None:
            self.adapted_ptm.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng theo `merge_mode` đã cấu hình."""
        with torch.no_grad():
            if self.merge_mode == "ptm_only":
                return _extract_features(self.frozen_ptm, x)
            if self.merge_mode == "adapted_only":
                return _extract_features(self.adapted_ptm, x)

            feat_adapted = _extract_features(self.adapted_ptm, x)
            feat_frozen = _extract_features(self.frozen_ptm, x)
            return torch.cat([feat_adapted, feat_frozen], dim=-1)