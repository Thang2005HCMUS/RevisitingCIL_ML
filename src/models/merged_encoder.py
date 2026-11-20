import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _validate_alpha(alpha: float) -> float:
    """Validate that mixing coefficient alpha is a real number in [0.0, 1.0]."""
    if isinstance(alpha, bool):
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}.")
    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}.") from exc
    if math.isnan(alpha_val) or not (0.0 <= alpha_val <= 1.0):
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}.")
    return alpha_val


def _resolve_projected_dim(
    projector: Optional[Callable],
    projected_dim: Optional[int],
    base_out_dim: int,
) -> int:
    """Determine output dimension after optional projection."""
    if projected_dim is not None:
        if isinstance(projected_dim, bool):
            raise ValueError(f"projected_dim must be a positive integer, got {projected_dim}.")
        dim_val = int(projected_dim)
        if dim_val <= 0:
            raise ValueError(f"projected_dim must be a positive integer, got {projected_dim}.")
        return dim_val

    if projector is not None:
        for attr in ("out_dim", "n_components_", "n_components"):
            val = getattr(projector, attr, None)
            if val is None or isinstance(val, bool):
                continue
            if attr == "n_components" and isinstance(val, float) and val < 1.0:
                continue
            if isinstance(val, (int, float)) or type(val).__name__.startswith("int"):
                dim_val = int(val)
                if dim_val > 0:
                    return dim_val
    return base_out_dim


class DualBranchEncoder(nn.Module):
    """
    Module hợp nhất đặc trưng từ mô hình PTM gốc và mô hình đã thích ứng (Adapted).
    Hỗ trợ 4 chế độ ghép (`merge_mode`): "both", "adapted_only", "ptm_only", "dual_ptm",
    cùng tùy chọn pha trộn có trọng số (`alpha`, `weighted_norm`) và phép chiếu (`projector`).
    """

    def __init__(
        self,
        frozen_ptm: Optional[nn.Module],
        adapted_ptm: Optional[nn.Module],
        merge_mode: str = "both",
        alpha: float = 0.5,
        weighted_norm: bool = False,
        projector: Optional[Callable] = None,
        projected_dim: Optional[int] = None,
    ):
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

        if projector is not None and not callable(projector) and not (
            hasattr(projector, "transform") and callable(projector.transform)
        ):
            raise TypeError("projector must be callable or expose a .transform() method.")

        self.merge_mode = mode
        self.frozen_ptm = frozen_ptm
        self.adapted_ptm = adapted_ptm
        self.alpha = _validate_alpha(alpha)
        self.weighted_norm = bool(weighted_norm)
        self.projector = projector

        if self.merge_mode == "ptm_only":
            base_out_dim = _get_module_out_dim(self.frozen_ptm)
        elif self.merge_mode == "adapted_only":
            base_out_dim = _get_module_out_dim(self.adapted_ptm)
        else:
            base_out_dim = _get_module_out_dim(self.adapted_ptm) + _get_module_out_dim(self.frozen_ptm)

        self.out_dim = _resolve_projected_dim(self.projector, projected_dim, base_out_dim)

        # Đảm bảo các nhánh ở trạng thái eval và đóng băng gradient.
        if self.frozen_ptm is not None:
            self.frozen_ptm.eval()
        if self.adapted_ptm is not None:
            self.adapted_ptm.eval()
        if isinstance(self.projector, nn.Module):
            self.projector.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _apply_projector(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply optional callable or .transform() projector and return a tensor on feat's device."""
        if hasattr(self.projector, "transform") and callable(self.projector.transform):
            projected = self.projector.transform(feat.detach().cpu().numpy())
        else:
            try:
                projected = self.projector(feat)
            except TypeError:
                projected = self.projector(feat.detach().cpu().numpy())

        if isinstance(projected, torch.Tensor):
            return projected.to(device=feat.device, dtype=feat.dtype)
        return torch.as_tensor(projected, device=feat.device, dtype=feat.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng theo `merge_mode`, `alpha` và `projector` đã cấu hình."""
        with torch.no_grad():
            if self.merge_mode == "ptm_only":
                feat = _extract_features(self.frozen_ptm, x)
                if self.weighted_norm:
                    feat = F.normalize(feat, p=2, dim=-1)
            elif self.merge_mode == "adapted_only":
                feat = _extract_features(self.adapted_ptm, x)
                if self.weighted_norm:
                    feat = F.normalize(feat, p=2, dim=-1)
            else:
                feat_adapted = _extract_features(self.adapted_ptm, x)
                feat_frozen = _extract_features(self.frozen_ptm, x)
                if self.weighted_norm:
                    feat_adapted = F.normalize(feat_adapted, p=2, dim=-1)
                    feat_frozen = F.normalize(feat_frozen, p=2, dim=-1)
                    feat = torch.cat(
                        [
                            math.sqrt(self.alpha) * feat_adapted,
                            math.sqrt(1.0 - self.alpha) * feat_frozen,
                        ],
                        dim=-1,
                    )
                else:
                    feat = torch.cat([feat_adapted, feat_frozen], dim=-1)

            if self.projector is not None:
                feat = self._apply_projector(feat)
            return feat