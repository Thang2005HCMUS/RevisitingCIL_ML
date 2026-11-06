import os
from pathlib import Path

# Danh sách toàn bộ các file rỗng cần khởi tạo theo cấu trúc dự án
PROJECT_FILES = [
    # Configs
    "configs/simplecil_cifar100.yaml",
    "configs/aper_adapter_imagenetr.yaml",
    
    # Source code - Data
    "src/data/__init__.py",
    "src/data/datasets.py",
    "src/data/task_manager.py",
    
    # Source code - Models
    "src/models/__init__.py",
    "src/models/backbones.py",
    "src/models/classifiers.py",
    "src/models/merged_encoder.py",
    
    # Source code - Tuning Strategies (Strategy Pattern)
    "src/tuning/__init__.py",
    "src/tuning/base.py",
    "src/tuning/adapter.py",
    "src/tuning/prompt.py",
    "src/tuning/scale_shift.py",
    "src/tuning/linear_probe.py",
    
    # Source code - Framework Core
    "src/framework/__init__.py",
    "src/framework/evaluator.py",
    "src/framework/learner.py",
    
    # Source code - Utils
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/metrics.py",
    
    # Experiments & Tests
    "experiments/.gitkeep",
    "tests/__init__.py",
    "tests/test_models.py",
    
    # Root files
    "README.md",
    "requirements.txt",
    "main.py",
]

def init_project_structure():
    base_path = Path.cwd()
    print(f"Bắt đầu khởi tạo cấu trúc dự án tại: {base_path}\n")

    for file_route in PROJECT_FILES:
        file_path = base_path / Path(file_route)
        
        # Tự động tạo toàn bộ thư mục cha nếu chưa tồn tại
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo file rỗng nếu chưa có
        if not file_path.exists():
            file_path.touch()
            print(f" [+] Đã tạo: {file_route}")
        else:
            print(f" [!] Bỏ qua (đã tồn tại): {file_route}")

    print("\nHoàn tất khởi tạo toàn bộ cấu trúc thư mục và file.")

if __name__ == "__main__":
    init_project_structure()