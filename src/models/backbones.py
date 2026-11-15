import torch
import torch.nn as nn
import timm
import logging

class SSFHeadWrapper(nn.Module):
    """Lớp bọc thêm Scale & Shift vào đầu ra của ResNet (tương đương resnet_scale.py)."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.out_dim = backbone.num_features
        self.ssf_scale_1 = nn.Parameter(torch.ones(self.out_dim))
        self.ssf_shift_1 = nn.Parameter(torch.zeros(self.out_dim))
        nn.init.normal_(self.ssf_scale_1, mean=1.0, std=0.02)
        nn.init.normal_(self.ssf_shift_1, std=0.02)

    def forward(self, x):
        features = self.backbone(x)
        return features * self.ssf_scale_1 + self.ssf_shift_1


def get_pretrained_model(args: dict) -> nn.Module:
    convnet_type = args.get("convnet_type", "vit_base_patch16_224_in21k").lower()
    logging.info(f"Đang tải backbone cơ sở cho: {convnet_type}")
    
    try:
        # 1. Xử lý riêng cho nhánh ResNet (BN Tuning / SSF / Finetune)
        if "resnet" in convnet_type:
            # Tách lấy tên gốc (vd: 'pretrained_resnet18_ssf' -> 'resnet18')
            base_name = convnet_type.replace("pretrained_", "").replace("_ssf", "").replace("_bn", "")
            base_model = timm.create_model(base_name, pretrained=True, num_classes=0)
            
            if "_ssf" in convnet_type:
                model = SSFHeadWrapper(base_model)
            else:
                model = base_model
                model.out_dim = base_model.num_features

        # 2. Các nhánh ViT giữ nguyên như cũ
        elif "_ssf" in convnet_type:
            import convs.vision_transformer_ssf 
            base_name = convnet_type.replace("pretrained_", "")
            if "vit_b16_224" in base_name:
                base_name = base_name.replace("vit_b16_224", "vit_base_patch16_224")
            model = timm.create_model(base_name, pretrained=True, num_classes=0)
            model.out_dim = 768
            
        elif "_adapter" in convnet_type:
            from convs import vision_transformer_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                ffn_adapt=True, ffn_option="parallel", ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora", ffn_adapter_scalar="0.1",
                ffn_num=args.get("ffn_num", 64), d_model=768, vpt_on=False, vpt_num=0
            )
            if "in21k" in convnet_type:
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(
                    num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            else:
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(
                    num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            model.out_dim = 768
            
        elif "_vpt" in convnet_type:
            from convs.vpt import build_promptmodel
            base_name = "vit_base_patch16_224_in21k" if "in21k" in convnet_type else "vit_base_patch16_224"
            vpt_type = "Deep" if args.get("vpt_type") == "deep" else "Shallow"
            prompt_num = args.get("prompt_token_num", 10)
            model = build_promptmodel(modelname=base_name, Prompt_Token_num=prompt_num, VPT_type=vpt_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim = 768
            
        else:
            base_name = convnet_type.replace("pretrained_", "")
            if "vit_b16_224" in base_name:
                base_name = base_name.replace("vit_b16_224", "vit_base_patch16_224")
            model = timm.create_model(base_name, pretrained=True, num_classes=0)
            model.out_dim = model.num_features if hasattr(model, 'num_features') else model.embed_dim

    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình '{convnet_type}': {e}")
        raise e
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    logging.info(f"Đã tải thành công với feature_dim = {model.out_dim}")
    return model