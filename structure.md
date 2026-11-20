### 1. Cấu trúc thư mục dự án (Project Layout)

```text
RevisitingCIL_ML/
│
├── main.py                           # Entry point: setup_parser(), load_json(), main() -> train()
├── run_exp3_sensitivity.py           # Entry point Exp 3: 2D alpha-mixing & Stage-1 PCA sensitivity sweep
├── download_model.py                 # Tải các pretrained backbones từ model.txt vào pretrained_models/
├── download_dataset.py               # Tải dataset từ dataset.txt (hỗ trợ GDrive/SharePoint, kiểm tra HTML lỗi)
├── extract_zip.py                    # Giải nén các file .zip theo cấu hình extract_zip.txt
├── dataset.txt / full_dataset.txt    # Danh sách URL tải datasets (CUB, IN-R, IN-A, OmniBenchmark, VTAB, ObjectNet)
├── extract_zip.txt / model.txt       # Danh sách file nén và URL pretrained weights
├── SimpleCIL.ipynb                   # Notebook chạy thực nghiệm trên Google Colab
├── README.md                         # Hướng dẫn cài đặt, W&B API key và chạy lệnh trên Colab
├── requirements.txt / setup.py
│
├── exps/                             # Cấu hình JSON chuẩn cho các thí nghiệm Table 1 & ResNet
│   ├── simplecil/                    # Baseline đóng băng hoàn toàn (7 datasets)
│   ├── aper_finetune/                # APER w/ Finetune (7 datasets)
│   ├── aper_vpt-shallow/             # APER w/ VPT-Shallow (7 datasets)
│   ├── aper_vpt-deep/                # APER w/ VPT-Deep (7 datasets)
│   ├── aper_ssf/                     # APER w/ SSF (7 datasets + nhánh resnet/)
│   ├── aper_adapter/                 # APER w/ Adapter (7 datasets)
│   ├── aper_bn/                      # APER w/ BN Tuning cho ResNet18 (cifar224, omni)
│   └── exp3_analysis/                # Cấu hình JSON quét lưới 2D (alpha, k) cho Exp 3 (cifar224, imagenetr, cub)
│
├── configs/                          # Cấu hình YAML mẫu (simplecil_cifar100.yaml, aper_adapter_imagenetr.yaml)
├── experiments/                      # Thư mục lưu kết quả thực nghiệm
│
├── src/                              # Mã nguồn chính
│   ├── data/                         # Quản lý dữ liệu & phân chia CIL tasks
│   │   ├── datasets.py               # ContinualDataset, build_transform(), get_dataset() (CIFAR224, CUB, IN-R, IN-A, OMNI, VTAB)
│   │   └── task_manager.py           # ContinualTaskManager: chia B tasks, get_task_loaders(), get_class_ids()
│   │
│   ├── convs/                        # Định nghĩa kiến trúc backbone tích hợp PET (tương thích timm >= 1.0)
│   │   ├── vision_transformer_adapter.py  # ViT + AdaptMLP (vit_base_patch16_224[_in21k]_adapter)
│   │   ├── vision_transformer_ssf.py      # ViT + Scale & Shift (_create_vision_transformer)
│   │   ├── vpt.py                         # ViT + Visual Prompt Tuning (build_promptmodel)
│   │   ├── new_adapter.py / new_ssf.py    # Bản viết lại cấu trúc Adapter & SSF
│   │   └── *.txt                          # Mã tham chiếu kiến trúc ResNet/linears gốc
│   │
│   ├── models/                       # Khởi tạo backbone, hợp nhất nhánh & bộ phân loại
│   │   ├── backbones.py              # get_pretrained_model(), SSFHeadWrapper (ResNet SSF)
│   │   ├── classifiers.py            # CosinePrototypeClassifier: register_prototypes(), forward()
│   │   └── merged_encoder.py         # DualBranchEncoder: ghép đặc trưng [phi*(x), phi(x)]
│   │
│   ├── tuning/                       # Strategy Pattern cho các kỹ thuật tinh chỉnh (PET)
│   │   ├── __init__.py               # Factory get_tuning_strategy(convnet_type, model_name)
│   │   ├── base.py                   # TuningStrategy (ABC): requires_training, is_forward_only, prepare_adapted_model()
│   │   ├── linear_probe.py           # FreezeStrategy (SimpleCIL)
│   │   ├── finetune.py               # FinetuneStrategy (APER w/ Finetune)
│   │   ├── bn.py                     # BNTuningStrategy (Forward-only cập nhật running stats)
│   │   ├── prompt.py                 # PromptStrategy (APER w/ VPT Shallow/Deep)
│   │   ├── scale_shift.py            # SSFStrategy (APER w/ SSF)
│   │   └── adapter.py                # AdapterStrategy (APER w/ Adapter)
│   │
│   ├── framework/                    # Luồng điều phối huấn luyện, đánh giá & W&B tracking
│   │   ├── learner.py                # train(), _train(), _init_wandb(), APERLearner
│   │   └── evaluator.py              # evaluate(), _compute_metrics() (Top-1, Top-5, Old/New, Grouped)
│   │
│   ├── analysis/                     # Module phân tích mở rộng (Exp 3)
│   │   └── sensitivity_sweep.py      # extract_and_cache_features(), run_2d_sensitivity_sweep(), vẽ biểu đồ & log W&B
│   │
│   └── utils/                        # Tiện ích phụ trợ
│       ├── logger.py                 # setup_logger()
│       └── metrics.py                # count_parameters(), tensor2numpy(), target2onehot()
│
└── tests/                            # Unit tests
    ├── test_models.py                # Kiểm tra kiến trúc/forward model
    ├── test_wandb_tracking.py        # Kiểm tra metric Table 1 & tích hợp W&B
    └── test_exp3_sensitivity.py      # Kiểm tra Exp 3: alpha-mixing, caching, PCA SVD fallback, plots & W&B
```

---

### 2. Các thành phần & hàm quan trọng (Key Modules & Functions)

| Thành phần | File & Symbol chính | Vai trò chính |
|---|---|---|
| **CLI & Entrypoint** | `main.py`: `setup_parser()`, `main()` | Đọc config JSON từ `--config`, nhận cờ `--wandb`, `--wandb_entity` (`nnm-hcmus`), `--wandb_project` (`Group7-ML-Lab`), rồi gọi `train(args)`. |
| **Data & CIL Split** | `src/data/datasets.py`: `get_dataset()`<br>`src/data/task_manager.py`: `ContinualTaskManager` | Nạp dataset (`cifar224`, `cub`, `imagenetr`, `imageneta`, `omnibenchmark`, `vtab`), xáo trộn thứ tự lớp theo `seed` và tạo `train_loader`, `test_loader`, `proto_loader` cho từng task. |
| **Backbone & Merge** | `src/models/backbones.py`: `get_pretrained_model()`<br>`src/models/merged_encoder.py`: `DualBranchEncoder`<br>`src/models/classifiers.py`: `CosinePrototypeClassifier` | Nạp PTM gốc (`ViT` hoặc `ResNet`) và đóng băng mặc định; nối đặc trưng 2 nhánh $[\phi^*(x), \phi(x)]$; tính Cosine Similarity với trọng số nguyên mẫu `register_prototypes()`. |
| **Tuning Strategies** | `src/tuning/__init__.py`: `get_tuning_strategy()`<br>`FreezeStrategy`, `FinetuneStrategy`, `BNTuningStrategy`, `PromptStrategy`, `SSFStrategy`, `AdapterStrategy` | Chọn chiến lược theo `(convnet_type, model_name)` và bật `requires_grad` đúng nhóm tham số PET (hoặc reset BN stats với BN Tuning) trước khi thích ứng ở Task 0. |
| **Learner & W&B** | `src/framework/learner.py`: `train()`, `_train()`, `_init_wandb()`, `_resolve_method_name()`, `APERLearner` | Điều phối vòng lặp $B$ task (`execute_task`, `_adapt_model`, `_update_prototypes`, `evaluate`), tính chỉ số Table 1 ($\bar{\mathcal{A}}$ `avg_acc`, $\mathcal{A}_B$ `last_acc`) và log lên W&B. |
| **Evaluator** | `src/framework/evaluator.py`: `evaluate()`, `_compute_metrics()` | Tính `top1` ($\mathcal{A}_b$), `top5`, `old`, `new` và `grouped` accuracy trên các lớp đã học. |
| **Exp 3 Sweep** | `run_exp3_sensitivity.py`<br>`src/analysis/sensitivity_sweep.py`: `extract_and_cache_features()`, `run_2d_sensitivity_sweep()`, `generate_sensitivity_plots()` | Cache đặc trưng hai nhánh sau Stage 0, quét lưới $(\alpha, k)$ với PCA / Random Subsampling, xuất CSV/PNG và đẩy bảng/biểu đồ lên W&B. |
| **Download Guard** | `download_dataset.py`: `download_datasets()`, `is_html_error_file()` | Tải dữ liệu qua Google Drive/OneDrive và tự xóa file `.zip` giả nếu bị trả về trang HTML lỗi (hết quota). |

---

### 3. Luồng thực thi tổng quát (Workflow)

```text
main.py (--config exps/...json [--wandb])
   │
   ▼
train(args) -> _train(args)
   ├─ ContinualTaskManager(args): chia C lớp thành B tasks (init_cls + n * (B-1))
   ├─ APERLearner(args): tạo frozen_ptm (phi) + adapted_ptm (phi*) theo TuningStrategy
   └─ _init_wandb(args, learner): khởi tạo W&B run (nnm-hcmus/Group7-ML-Lab) nếu bật --wandb
   │
   ▼
Vòng lặp Task 0 .. B-1:
   ├─ Task 0:
   │    ├─ _adapt_model(train_loader): bỏ qua (SimpleCIL) / forward-only (BN) / train tuned_epoch (APER)
   │    └─ Đóng băng adapted_ptm -> ghép DualBranchEncoder([phi*(x), phi(x)])
   ├─ Mọi Task b (0 .. B-1):
   │    ├─ _update_prototypes(proto_loader, class_ids): trích mean embedding -> CosinePrototypeClassifier
   │    └─ evaluate(test_loader, known_classes, increment): tính A_b (Top-1, Top-5, Old, New) + log W&B
   │
   ▼
Kết thúc B tasks:
   └─ Tổng hợp & log chỉ số Table 1: avg_acc (mean A_1..A_B), last_acc (A_B) -> đóng wandb.finish()
```
