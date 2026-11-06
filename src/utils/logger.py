import os
import sys
import logging

def setup_logger(args: dict):
    """
    Thiết lập logging cho dự án. Ghi log ra cả console và file.
    Đường dẫn file log được tạo tự động dựa trên cấu hình (dataset, model, task).
    """
    # Xác định số lớp ban đầu
    init_cls = 0 if args.get("init_cls") == args.get("increment") else args.get("init_cls", 0)
    
    # Tạo thư mục lưu log
    log_dir = f"logs/{args.get('model_name', 'aper')}/{args.get('dataset', 'unknown')}/{init_cls}/{args.get('increment', 0)}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Tạo tên file log
    logfilename = f"{log_dir}/{args.get('prefix', 'exp')}_{args.get('seed', 1)}_{args.get('convnet_type', 'vit')}.log"
    
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logging.info(f"Đã khởi tạo Logger. File log lưu tại: {logfilename}")