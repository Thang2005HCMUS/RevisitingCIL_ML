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
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases experiment tracking.')
    parser.add_argument('--wandb_entity', type=str, default='nnm-hcmus',
                        help='Weights & Biases entity (team or username).')
    parser.add_argument('--wandb_project', type=str, default='Group7-ML-Lab',
                        help='Weights & Biases project name.')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='Weights & Biases run group.')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name.')
    return parser

def main():
    # Phân tích tham số từ dòng lệnh
    args = setup_parser().parse_args()
    
    # Tải cấu hình từ file JSON 
    param = load_json(args.config)
    
    # Chuyển đổi argparse.Namespace thành dictionary và cập nhật với các tham số từ JSON
    cli_args = vars(args)
    args_dict = dict(cli_args)
    args_dict.update(param)
    if cli_args.get('wandb'):
        args_dict['wandb'] = True
    if cli_args.get('wandb_group') is not None:
        args_dict['wandb_group'] = cli_args['wandb_group']
    if cli_args.get('wandb_name') is not None:
        args_dict['wandb_name'] = cli_args['wandb_name']
    args_dict['use_wandb'] = bool(args_dict.get('wandb') or args_dict.get('use_wandb'))
    
    # Truyền bộ args cuối cùng vào luồng huấn luyện chính
    train(args_dict)

if __name__ == '__main__':
    main()