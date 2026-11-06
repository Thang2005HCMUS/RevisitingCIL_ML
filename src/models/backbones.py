import torch
import torch.nn as nn
import timm
import logging

def get_pretrained_model(convnet_type: str) -> nn.Module:
    """
    Hàm Factory khởi tạo và trả về mô hình tiền huấn luyện (PTM) gốc.
    Các chiến lược tinh chỉnh (Adapter, SSF, VPT) sẽ được cấy vào sau bởi TuningStrategy.
  
    """
    convnet_type = convnet_type.lower()
    logging.info(f"Đang tải backbone cơ sở cho: {convnet_type}")
    
    # 1. Bóc tách tên mô hình thực sự (Loại bỏ các hậu tố do code cũ đặt)
    # Ví dụ: 'pretrained_vit_b16_224_adapter' -> 'vit_base_patch16_224'
    base_name = convnet_type.replace("pretrained_", "")
    base_name = base_name.replace("_ssf", "").replace("_vpt", "").replace("_adapter", "")
    
    # Chuẩn hóa lại tên cho thư viện timm (Ví dụ: vit_b16_224 -> vit_base_patch16_224)
    if "vit_b16_224" in base_name:
        base_name = base_name.replace("vit_b16_224", "vit_base_patch16_224")

    try:
        # 2. Tải mô hình từ timm với num_classes=0 (Chỉ lấy Feature Extractor, bỏ lớp Head)
        model = timm.create_model(base_name, pretrained=True, num_classes=0)
        
        # 3. Đồng bộ hóa thuộc tính out_dim để Classifier phía sau biết kích thước
        if hasattr(model, 'num_features'):
            model.out_dim = model.num_features
        elif hasattr(model, 'embed_dim'):
            model.out_dim = model.embed_dim
        else:
            # Fallback cho các mạng như ResNet
            dummy_input = torch.randn(1, 3, 224, 224)
            out = model(dummy_input)
            model.out_dim = out.shape[1]
            
    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình '{base_name}' từ timm: {e}")
        raise e
        
    # Đóng băng toàn bộ (sẽ được mở khóa chọn lọc tại TuningStrategy)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    logging.info(f"Đã tải thành công {base_name} với feature_dim = {model.out_dim}")
    return model