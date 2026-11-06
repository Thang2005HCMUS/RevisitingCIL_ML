import json
import argparse
import os
import sys

# Thêm thư mục 'src' vào đường dẫn hệ thống để import các module theo cấu trúc mới
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Dựa trên cấu trúc mới, luồng huấn luyện sẽ được chuyển vào src/framework/
# Bạn có thể import hàm khởi chạy (ví dụ: train) từ framework thay vì trainer.py như code gốc
from framework.learner import train 

def load_json(settings_path):
    """
    Đọc cấu hình từ file JSON.
    Giữ nguyên logic của tác giả để tương thích với các file JSON cũ.
    """
    with open(settings_path, 'r', encoding='utf-8') as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    """Thiết lập các tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Reproduce of APER and multiple continual learning algorithms.')
    # Trỏ mặc định về thư mục configs/ theo cấu trúc mới
    parser.add_argument('--config', type=str, default='./configs/simplecil_cifar100.json',
                        help='Json file of settings.')
    return parser

def main():
    # Phân tích tham số từ dòng lệnh
    args = setup_parser().parse_args()
    
    # Tải cấu hình từ file JSON 
    param = load_json(args.config)
    
    # Chuyển đổi argparse.Namespace thành dictionary và cập nhật với các tham số từ JSON
    args_dict = vars(args)
    args_dict.update(param)
    
    # Truyền bộ args cuối cùng vào luồng huấn luyện chính
    train(args_dict)

if __name__ == '__main__':
    main()