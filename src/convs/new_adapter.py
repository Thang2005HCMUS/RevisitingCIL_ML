import math
from functools import partial
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, PatchEmbed


class BottleneckAdapter(nn.Module):
    """Nhánh phụ dạng thắt cổ chai (Down -> ReLU -> Dropout -> Up) theo kiến trúc AdaptFormer."""
    def __init__(self, embed_dim: int, bottleneck_dim: int, dropout: float = 0.1,
                 scale_mode: str = "0.1", norm_mode: str = "none"):
        super().__init__()
        self.norm_mode = norm_mode
        self.pre_norm = nn.LayerNorm(embed_dim) if norm_mode in ("in", "out") else nn.Identity()

        if scale_mode == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(scale_mode)

        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, add_residual: bool = True, residual: torch.Tensor = None) -> torch.Tensor:
        skip = x if residual is None else residual
        h = self.pre_norm(x) if self.norm_mode == "in" else x

        h = self.up_proj(self.drop(self.act(self.down_proj(h)))) * self.scale

        if self.norm_mode == "out":
            h = self.pre_norm(h)

        return (h + skip) if add_residual else h


class MultiHeadSelfAttention(nn.Module):
    """Khối Self-Attention với 3 lớp chiếu Q, K, V độc lập."""
    def __init__(self, embed_dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Tách thành [B, num_heads, N, head_dim]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = self.attn_drop(F.softmax(attn_scores, dim=-1))

        out = (attn_probs @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerEncoderLayer(nn.Module):
    """Một tầng Transformer Block tích hợp tùy chọn AdaptMLP (song song hoặc nối tiếp)."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, tuning_cfg=None):
        super().__init__()
        self.tuning_cfg = tuning_cfg
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

        if self.tuning_cfg is not None and self.tuning_cfg.ffn_adapt:
            self.adaptmlp = BottleneckAdapter(
                embed_dim=embed_dim,
                bottleneck_dim=self.tuning_cfg.ffn_num,
                dropout=0.1,
                scale_mode=self.tuning_cfg.ffn_adapter_scalar,
                norm_mode=self.tuning_cfg.ffn_adapter_layernorm_option,
            )

    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp_drop(self.act(self.fc1(x)))
        return self.drop_path(self.mlp_drop(self.fc2(h)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))

        residual = x
        normed_x = self.norm2(x)
        mlp_out = self._forward_mlp(normed_x)

        if self.tuning_cfg is not None and self.tuning_cfg.ffn_adapt:
            mode = self.tuning_cfg.ffn_option
            if mode == "parallel":
                mlp_out = mlp_out + self.adaptmlp(x, add_residual=False)
            elif mode == "sequential":
                mlp_out = self.adaptmlp(mlp_out)
            else:
                raise ValueError(f"Chế độ ffn_option không hợp lệ: {mode}")

        return residual + mlp_out


class VisionTransformerAdapter(nn.Module):
    """Kiến trúc Vision Transformer hỗ trợ cắm các mô-đun Adapter."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 num_classes: int = 0, embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                 norm_layer=None, act_layer=None, tuning_config=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_fn = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_fn = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_rates = [r.item() for r in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_rates[i],
                act_layer=act_fn, norm_layer=norm_fn, tuning_cfg=tuning_config
            )
            for i in range(depth)
        ])

        self.norm = norm_fn(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self.patch_embed(x)
        cls_tok = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for layer in self.blocks:
            tokens = layer(tokens)

        tokens = self.norm(tokens)
        return tokens[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def _convert_timm_state_dict(timm_state_dict: dict) -> dict:
    """Chuyển đổi tên tham số từ checkpoint của timm sang cấu trúc Q/K/V tách rời."""
    converted = {}
    for k, v in timm_state_dict.items():
        if "qkv.weight" in k:
            q, k_w, v_w = torch.chunk(v, 3, dim=0)
            converted[k.replace("qkv.weight", "q_proj.weight")] = q
            converted[k.replace("qkv.weight", "k_proj.weight")] = k_w
            converted[k.replace("qkv.weight", "v_proj.weight")] = v_w
        elif "qkv.bias" in k:
            q, k_b, v_b = torch.chunk(v, 3, dim=0)
            converted[k.replace("qkv.bias", "q_proj.bias")] = q
            converted[k.replace("qkv.bias", "k_proj.bias")] = k_b
            converted[k.replace("qkv.bias", "v_proj.bias")] = v_b
        elif "mlp.fc" in k:
            converted[k.replace("mlp.", "")] = v
        else:
            converted[k] = v
    return converted


def _create_adapted_vit(model_name: str, **kwargs) -> VisionTransformerAdapter:
    kwargs.pop("global_pool", None)

    model = VisionTransformerAdapter(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    ref_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    mapped_state = _convert_timm_state_dict(ref_model.state_dict())
    incompatible = model.load_state_dict(mapped_state, strict=False)

    # Đóng băng toàn bộ backbone gốc, chỉ cho phép huấn luyện các tham số mới (Adapter)
    trainable_keys = set(incompatible.missing_keys)
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_keys

    return model


def vit_base_patch16_224_adapter(pretrained: bool = False, **kwargs) -> VisionTransformerAdapter:
    return _create_adapted_vit("vit_base_patch16_224", **kwargs)


def vit_base_patch16_224_in21k_adapter(pretrained: bool = False, **kwargs) -> VisionTransformerAdapter:
    return _create_adapted_vit("vit_base_patch16_224_in21k", **kwargs)