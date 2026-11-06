import torch
import torch.nn as nn
import timm
import logging

def get_pretrained_model(args: dict) -> nn.Module:
    """
    Hàm Factory khởi tạo và trả về mô hình tiền huấn luyện (PTM) gốc.
    """
    convnet_type = args.get("convnet_type", "vit_base_patch16_224_in21k").lower()
    logging.info(f"Đang tải backbone cơ sở cho: {convnet_type}")
    
    try:
        if "_ssf" in convnet_type:
            # Import từ thư mục src.convs vừa được đưa vào
            import convs.vision_transformer_ssf 
            base_name = convnet_type.replace("pretrained_", "")
            if "vit_b16_224" in base_name:
                base_name = base_name.replace("vit_b16_224", "vit_base_patch16_224")
            
            # Lúc này timm đã có thể tạo model CÓ CÁC THAM SỐ SSF nhờ @register_model của tác giả[cite: 3]
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
            # Mô hình ViT chuẩn (SimpleCIL / Finetune) dùng trực tiếp timm
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