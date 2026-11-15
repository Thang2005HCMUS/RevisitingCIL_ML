from functools import partial
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_, to_2tuple
from timm.models import register_model


def create_ssf_params(num_features: int):
    """Khởi tạo cặp tham số Scale (gamma ~ N(1, 0.02)) và Shift (beta ~ N(0, 0.02))."""
    gamma = nn.Parameter(torch.ones(num_features))
    beta = nn.Parameter(torch.zeros(num_features))
    nn.init.normal_(gamma, mean=1.0, std=0.02)
    nn.init.normal_(beta, std=0.02)
    return gamma, beta


def apply_ssf(tensor: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Thực hiện phép biến đổi affine theo từng kênh: x * gamma + beta."""
    if tensor.shape[-1] == gamma.shape[0]:
        return tensor * gamma + beta
    if tensor.shape[1] == gamma.shape[0]:
        return tensor * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    raise ValueError(f"Kích thước tensor {tensor.shape} không khớp với tham số SSF {gamma.shape}")


class ImagePatchEmbedding(nn.Module):
    """Chia ảnh 2D thành chuỗi các vector nhúng (Patch Tokens) có hỗ trợ SSF."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, tuning_mode="ssf"):
        super().__init__()
        img_hw = to_2tuple(img_size)
        patch_hw = to_2tuple(patch_size)
        self.grid_size = (img_hw[0] // patch_hw[0], img_hw[1] // patch_hw[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.use_ssf = (tuning_mode == "ssf")

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        if self.use_ssf:
            self.ssf_scale_1, self.ssf_shift_1 = create_ssf_params(embed_dim)
            if norm_layer is not None:
                self.ssf_scale_2, self.ssf_shift_2 = create_ssf_params(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.use_ssf:
            x = apply_ssf(x, self.ssf_scale_1, self.ssf_shift_1)
            x = self.norm(x)
            if hasattr(self, "ssf_scale_2"):
                x = apply_ssf(x, self.ssf_scale_2, self.ssf_shift_2)
        else:
            x = self.norm(x)
        return x


class SSFSelfAttention(nn.Module):
    """Khối Multi-Head Self-Attention chèn SSF sau phép chiếu QKV và Output Projection."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, tuning_mode: str = "ssf"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_ssf = (tuning_mode == "ssf")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_ssf:
            self.ssf_scale_1, self.ssf_shift_1 = create_ssf_params(dim * 3)
            self.ssf_scale_2, self.ssf_shift_2 = create_ssf_params(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_out = self.qkv(x)
        if self.use_ssf:
            qkv_out = apply_ssf(qkv_out, self.ssf_scale_1, self.ssf_shift_1)

        qkv = qkv_out.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        if self.use_ssf:
            out = apply_ssf(out, self.ssf_scale_2, self.ssf_shift_2)

        return self.proj_drop(out)


class SSFFeedForward(nn.Module):
    """Mạng MLP 2 lớp chèn SSF sau từng lớp Linear."""
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU,
                 drop: float = 0.0, tuning_mode: str = "ssf"):
        super().__init__()
        self.use_ssf = (tuning_mode == "ssf")
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

        if self.use_ssf:
            self.ssf_scale_1, self.ssf_shift_1 = create_ssf_params(hidden_features)
            self.ssf_scale_2, self.ssf_shift_2 = create_ssf_params(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.use_ssf:
            x = apply_ssf(x, self.ssf_scale_1, self.ssf_shift_1)
        x = self.drop1(self.act(x))

        x = self.fc2(x)
        if self.use_ssf:
            x = apply_ssf(x, self.ssf_scale_2, self.ssf_shift_2)
        return self.drop2(x)


class SSFTransformerBlock(nn.Module):
    """Một tầng Encoder Block hoàn chỉnh có chèn SSF ngay sau mỗi lớp LayerNorm."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, tuning_mode: str = "ssf"):
        super().__init__()
        self.use_ssf = (tuning_mode == "ssf")

        self.norm1 = norm_layer(dim)
        self.attn = SSFSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=drop, tuning_mode=tuning_mode
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = SSFFeedForward(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop, tuning_mode=tuning_mode
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.use_ssf:
            self.ssf_scale_1, self.ssf_shift_1 = create_ssf_params(dim)
            self.ssf_scale_2, self.ssf_shift_2 = create_ssf_params(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.norm1(x)
        if self.use_ssf:
            h1 = apply_ssf(h1, self.ssf_scale_1, self.ssf_shift_1)
        x = x + self.drop_path1(self.attn(h1))

        h2 = self.norm2(x)
        if self.use_ssf:
            h2 = apply_ssf(h2, self.ssf_scale_2, self.ssf_shift_2)
        x = x + self.drop_path2(self.mlp(h2))
        return x


class VisionTransformerSSF(nn.Module):
    """Backbone Vision Transformer tích hợp Scale & Shift (SSF) trên toàn mạng."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0,
                 global_pool="token", embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=None, act_layer=None, tuning_mode="ssf"):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.use_ssf = (tuning_mode == "ssf")

        norm_fn = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_fn = act_layer or nn.GELU

        self.patch_embed = ImagePatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, tuning_mode=tuning_mode
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_ssf:
            self.ssf_scale_1, self.ssf_shift_1 = create_ssf_params(embed_dim)

        drop_rates = [r.item() for r in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            SSFTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_rates[i],
                act_layer=act_fn, norm_layer=norm_fn, tuning_mode=tuning_mode
            )
            for i in range(depth)
        ])

        self.norm = norm_fn(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self.patch_embed(x)
        cls_tok = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        if self.use_ssf:
            tokens = apply_ssf(tokens, self.ssf_scale_1, self.ssf_shift_1)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features(x)
        feat = tokens[:, 1:].mean(dim=1) if self.global_pool == "avg" else tokens[:, 0]
        return self.head(feat)


def _build_ssf_vit(timm_arch: str, pretrained: bool = False, **kwargs) -> VisionTransformerSSF:
    """Khởi tạo VisionTransformerSSF và nạp trực tiếp trọng số từ timm bản mới."""
    # Loại bỏ các tham số nội bộ của timm factory nếu có
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    kwargs.pop("cache_dir", None)

    model = VisionTransformerSSF(
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    if pretrained:
        ref_model = timm.create_model(timm_arch, pretrained=True, num_classes=0)
        state_dict = {k: v for k, v in ref_model.state_dict().items() if "pre_logits" not in k}
        model.load_state_dict(state_dict, strict=False)

    return model


@register_model
def vit_tiny_patch16_224_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_tiny_patch16_224", pretrained=pretrained,
                          patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)


@register_model
def vit_small_patch16_224_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_small_patch16_224", pretrained=pretrained,
                          patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)


@register_model
def vit_base_patch16_224_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_base_patch16_224", pretrained=pretrained,
                          patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


@register_model
def vit_large_patch16_224_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_large_patch16_224", pretrained=pretrained,
                          patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)


@register_model
def vit_tiny_patch16_224_in21k_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_tiny_patch16_224_in21k", pretrained=pretrained,
                          patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)


@register_model
def vit_small_patch16_224_in21k_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_small_patch16_224_in21k", pretrained=pretrained,
                          patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)


@register_model
def vit_base_patch16_224_in21k_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_base_patch16_224_in21k", pretrained=pretrained,
                          patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


@register_model
def vit_large_patch16_224_in21k_ssf(pretrained: bool = False, **kwargs):
    return _build_ssf_vit("vit_large_patch16_224_in21k", pretrained=pretrained,
                          patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)