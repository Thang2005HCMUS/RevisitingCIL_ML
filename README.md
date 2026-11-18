# RevisitingCIL_ML

Repository tái hiện kết quả Table 1 cho `SimpleCIL` và các biến thể `APER` (`Finetune`, `VPT-Shallow`, `VPT-Deep`, `SSF`, `Adapter`, `BN`), tích hợp ghi log thực nghiệm lên Weights & Biases (`nnm-hcmus/Group7-ML-Lab`).

---

## Hướng dẫn chạy trên Google Colab (kèm W&B API Key)

### Bước 1: Bật GPU & Clone repository

Trong Google Colab, chọn **Runtime → Change runtime type → T4 GPU** (hoặc L4/A100), sau đó chạy cell:

```python
!git clone https://github.com/Thang2005HCMUS/RevisitingCIL_ML.git
%cd RevisitingCIL_ML
!pip -q install wandb timm easydict
```

### Bước 2: Nhét W&B API Key vào Colab

Lấy API key của bạn tại: <https://wandb.ai/authorize>

#### Cách A (Khuyên dùng — an toàn, không lộ key trong notebook): Dùng **Colab Secrets**
1. Bấm biểu tượng **chìa khóa (Secrets)** ở thanh bên trái Colab.
2. Tạo secret mới tên `WANDB_API_KEY`, dán key của bạn vào Value và bật toggle **Notebook access**.
3. Chạy cell sau:

```python
import os
import wandb
from google.colab import userdata

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
wandb.login(key=os.environ["WANDB_API_KEY"])
```

#### Cách B (Nhanh — dán trực tiếp vào cell nếu dùng cá nhân):

```python
import os
import wandb

os.environ["WANDB_API_KEY"] = "DÁN_WANDB_API_KEY_CỦA_BẠN_VÀO_ĐÂY"
wandb.login(key=os.environ["WANDB_API_KEY"])
```

> **Lưu ý:** Tránh commit notebook lên GitHub nếu bạn dùng Cách B (dán key trực tiếp trong cell).

### Bước 3: Tải pretrained models và datasets

```python
!python download_model.py model.txt
!python download_dataset.py dataset.txt
!python extract_zip.py extract_zip.txt
```

*(Nếu chỉ chạy một dataset cụ thể như CIFAR-100 hoặc ImageNet-R, có thể giải nén riêng file tương ứng trong `./data/`.)*

### Bước 4: Chạy thí nghiệm và log lên W&B (`nnm-hcmus/Group7-ML-Lab`)

Thêm cờ `--wandb` khi gọi `main.py` (mặc định sẽ log vào entity `nnm-hcmus` và project `Group7-ML-Lab`):

```bash
# SimpleCIL (CIFAR-224 B0 Inc5)
!python main.py --config ./exps/simplecil/cifar224_5_5_vit-b_simplecil.json --wandb

# APER Finetune (CUB-200 B0 Inc10)
!python main.py --config ./exps/aper_finetune/cub_10_10_vit-b_finetune.json --wandb

# APER VPT-Shallow & VPT-Deep (tự phân biệt method_name: aper_vpt-shallow vs aper_vpt-deep)
!python main.py --config ./exps/aper_vpt-shallow/cifar224_5_5_vit-b_vpt-shallow.json --wandb
!python main.py --config ./exps/aper_vpt-deep/cifar224_5_5_vit-b_vpt-deep.json --wandb

# APER SSF / Adapter / BN
!python main.py --config ./exps/aper_ssf/cifar224_5_5_vit-b_ssf.json --wandb
!python main.py --config ./exps/aper_adapter/cifar224_5_5_vit-b_adapter.json --wandb
!python main.py --config ./exps/aper_bn/cifar224_5_5_resnet18_bn.json --wandb
```

### Bước 5: Chạy Thực nghiệm 2 (Figure 5d & Figure 5f) trên Colab

#### Nhánh 2A — Giải phẫu thành phần (`ImageNet-A B100 Inc5`, Figure 5d)

```bash
# 1. SimpleCIL-PTM (chỉ dùng nhánh PTM IN21K đóng băng, 768-d)
!python main.py --config ./exps/exp2_submodules/imageneta_100_5_simplecil_ptm.json --wandb

# 2. SimpleCIL-Adapted (chỉ dùng nhánh đã finetune, 768-d)
!python main.py --config ./exps/exp2_submodules/imageneta_100_5_simplecil_adapted.json --wandb

# 3. SimpleCIL-21K+1K (ghép 2 nhánh PTM IN21K + IN1K đóng băng, 1536-d)
!python main.py --config ./exps/exp2_submodules/imageneta_100_5_simplecil_21k_1k.json --wandb

# 4. APER w/ Finetune (hợp nhất nhánh Adapted + PTM IN21K, 1536-d)
!python main.py --config ./exps/exp2_submodules/imageneta_100_5_aper_finetune.json --wandb
```

#### Nhánh 2B — Khảo sát số chặng thích nghi (`CIFAR-224 B0 Inc10`, Figure 5f)

Sử dụng một file cấu hình duy nhất kết hợp cờ `--adapt_stages T` với $T \in \{0, 1, \dots, 10\}$:

```bash
# Adapt-0 (SimpleCIL, không tinh chỉnh)
!python main.py --config ./exps/exp2_adapt_stages/cifar224_10_10_vit-b_adapter.json --adapt_stages 0 --wandb

# Adapt-1 (APER w/ Adapter tiêu chuẩn)
!python main.py --config ./exps/exp2_adapt_stages/cifar224_10_10_vit-b_adapter.json --adapt_stages 1 --wandb

# Adapt-2..10 (ví dụ Adapt-3 và Adapt-10)
!python main.py --config ./exps/exp2_adapt_stages/cifar224_10_10_vit-b_adapter.json --adapt_stages 3 --wandb
!python main.py --config ./exps/exp2_adapt_stages/cifar224_10_10_vit-b_adapter.json --adapt_stages 10 --wandb
```

### Tùy chọn CLI bổ sung

- `--merge_mode <mode>`: Chọn chế độ hợp nhất đặc trưng (`both`, `adapted_only`, `ptm_only`, `dual_ptm`), ghi đè giá trị trong JSON.
- `--adapt_stages <T>`: Số chặng đầu tiên cho phép thích nghi (`0` để bỏ qua bước Adapt; mặc định `1`), ghi đè giá trị trong JSON.
- `--wandb_entity <entity>`: Đổi W&B entity (mặc định: `nnm-hcmus`).
- `--wandb_project <project>`: Đổi W&B project (mặc định: `Group7-ML-Lab`).
- `--wandb_group <group>`: Đặt tên group tùy chỉnh (mặc định: `<dataset>_init<init_cls>_inc<increment>`).
- `--wandb_name <name>`: Đặt tên run tùy chỉnh (mặc định: `<method_name>_<dataset>_B<init_cls>_Inc<increment>_seed<seed>`).
- Chạy **không** truyền `--wandb`: Pipeline vẫn chạy đầy đủ và in chỉ số Table 1 (`avg_acc`, `last_acc`) ra console/file log mà không import hay gọi `wandb`.
