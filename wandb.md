# Hướng dẫn sử dụng Weights & Biases (W&B) trên Google Colab cho `RevisitingCIL_ML`

Tài liệu này hướng dẫn từng bước từ lúc **tạo tài khoản / lấy API Key trên W&B**, **gắn key vào Google Colab**, cho đến **chạy các thí nghiệm `SimpleCIL` & `APER`** và **xem bảng kết quả chuẩn Table 1** trên W&B (`nnm-hcmus/Group7-ML-Lab`).

---

## 1. Tạo tài khoản & Chuẩn bị Project trên W&B

1. Truy cập <https://wandb.ai/site> và đăng ký / đăng nhập tài khoản (có thể đăng nhập nhanh bằng tài khoản GitHub hoặc Google).
2. Chọn chế độ lưu log:
   - **Dùng Project chung của nhóm (`nnm-hcmus/Group7-ML-Lab` — mặc định của repo):**
     - Đảm bảo tài khoản W&B của bạn đã được mời vào team `nnm-hcmus` có quyền ghi vào project `Group7-ML-Lab`.
     - Khi chạy lệnh chỉ cần thêm `--wandb`, hệ thống tự đẩy log đúng vào `nnm-hcmus/Group7-ML-Lab`.

---

## 2. Lấy W&B API Key

1. Đăng nhập vào <https://wandb.ai>, sau đó mở trực tiếp trang:
   - <https://wandb.ai/authorize>
   - *(Hoặc bấm Avatar góc phải trên $\rightarrow$ **User Settings** $\rightarrow$ kéo xuống mục **API Keys** $\rightarrow$ **Reveal** / **Copy**).*
2. Sao chép chuỗi API key (dạng chuỗi ký tự dài khoảng 40 ký tự). **Không chia sẻ hoặc đẩy chuỗi key này công khai lên GitHub.**

---

## 3. Gắn W&B API Key lên Google Colab

Mở notebook trên Google Colab, chọn GPU (T4 hoặc L4/A100, tốt nhất L4/A100 để không bị quá tải), rồi nạp key:
1. Ở thanh công cụ bên trái Colab, bấm biểu tượng **Chìa khóa (Secrets)**.
2. Bấm **Thêm giá trị bí mật mới**:
   - **Name**: `WANDB_API_KEY`
   - **Value**: Dán API key vừa copy ở Bước 2
   - Bật công tắc **Notebook access** cho notebook hiện tại.
3. Chạy cell sau để cài `wandb` và đăng nhập:

```python
!pip -q install wandb timm easydict
import os
import wandb
from google.colab import userdata
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
wandb.login(key=os.environ["WANDB_API_KEY"])
```

---

## 4. Clone repo, tải Pretrained Models & Datasets trên Colab

### 4.1. Clone đúng nhánh mã nguồn và vào thư mục dự án

```python
!git clone -b 23122010_dev https://github.com/Thang2005HCMUS/RevisitingCIL_ML.git
%cd RevisitingCIL_ML
```

### 4.2. Tải Pretrained Models (bắt buộc)

```python
!python download_model.py model.txt
```

### 4.3. Tải Datasets


- **Nếu muốn chạy đủ 6 bộ dữ liệu (`CUB`, `ImageNet-R`, `ImageNet-A`, `OmniBenchmark`, `VTAB`, `CIFAR-224`):**

```python
!python download_dataset.py dataset.txt
!python extract_zip.py extract_zip.txt
!cd data && unzip -q imagenet-r.zip -d imagenet-r
```

*(Lưu ý: trong `dataset.txt` có link `objectnet.zip` ~121GB; nếu bạn chưa chạy ObjectNet thì có thể ngắt hoặc bỏ dòng `objectnet.zip` trong `dataset.txt` trước khi chạy).*

---

## 5. Chạy thí nghiệm và ghi log lên W&B

Cấu trúc lệnh chung cho mọi cặp `(Phương pháp × Bộ dữ liệu)` trong thư mục `exps/`:

```bash
!python main.py --config <duong_dan_file_json> --wandb
```

### 5.1. Chạy thử nghiệm nhanh trên `CIFAR-224 B0 Inc5` (20 tasks)

```bash
# 1. SimpleCIL (đóng băng backbone, không train, chạy rất nhanh để kiểm tra luồng W&B)
!python main.py --config ./exps/simplecil/cifar224_5_5_vit-b_simplecil.json --wandb

# 2. APER w/ Adapter
!python main.py --config ./exps/aper_adapter/cifar224_5_5_vit-b_adapter.json --wandb

# 3. APER w/ SSF
!python main.py --config ./exps/aper_ssf/cifar224_5_5_vit-b_ssf.json --wandb

# 4. APER w/ VPT-Shallow & VPT-Deep (hệ thống tự tách method_name: aper_vpt-shallow vs aper_vpt-deep)
!python main.py --config ./exps/aper_vpt-shallow/cifar224_5_5_vit-b_vpt-shallow.json --wandb
!python main.py --config ./exps/aper_vpt-deep/cifar224_5_5_vit-b_vpt-deep.json --wandb

# 5. APER w/ Finetune
!python main.py --config ./exps/aper_finetune/cifar224_5_5_vit-b_finetune.json --wandb

# 6. APER w/ BN Tuning (ResNet18)
!python main.py --config ./exps/aper_bn/cifar224_5_5_resnet18_bn.json --wandb
```

### 5.2. Ví dụ chạy trên các bộ dữ liệu khác (Table 1)

```bash
# CUB-200 (B0 Inc10)
!python main.py --config ./exps/simplecil/cub_10_10_vit-b_simplecil.json --wandb
!python main.py --config ./exps/aper_adapter/cub_10_10_vit-b_adapter.json --wandb

# ImageNet-R (B0 Inc5)
!python main.py --config ./exps/aper_ssf/imagenetr_5_5_vit-b_ssf.json --wandb
!python main.py --config ./exps/aper_adapter/imagenetr_5_5_vit-b_adapter.json --wandb

# ImageNet-A (B0 Inc10)
!python main.py --config ./exps/aper_ssf/imageneta_10_10_vit-b_ssf.json --wandb

# OmniBenchmark (B0 Inc30) & VTAB (B0 Inc10)
!python main.py --config ./exps/aper_vpt-deep/omni_30_30_vit-b_vpt-deep.json --wandb
!python main.py --config ./exps/aper_vpt-shallow/vtab_10_10_vit-b_vpt-shallow.json --wandb
```

### 5.3. Tùy chỉnh nâng cao qua CLI

| Cờ CLI | Mặc định | Ý nghĩa |
|---|---|---|
| `--wandb` | `False` | Bật ghi log lên W&B (nếu bỏ cờ này, code vẫn chạy và in kết quả ra console/file log mà không gọi W&B). |
| `--wandb_entity` | `nnm-hcmus` | Đổi team hoặc username nhận log (ví dụ `--wandb_entity my_username`). |
| `--wandb_project` | `Group7-ML-Lab` | Đổi tên project trên W&B. |
| `--wandb_group` | `<dataset>_init<init_cls>_inc<increment>` | Gom nhóm các run cùng dataset split để so sánh ngang 6 phương pháp. |
| `--wandb_name` | `<method_name>_<dataset>_B<init_cls>_Inc<increment>_seed<seed>` | Đặt tên hiển thị riêng cho run hiện tại. |
| `--wandb_tags` | `["backbone:<backbone_type>"]` | Gắn thêm tags bổ sung trên W&B (tự động gắn sẵn tag `backbone:<backbone_type>` và log cột `backbone_type`). |

Ví dụ log thử vào project cá nhân với tên run tùy chỉnh:

```bash
!python main.py --config ./exps/simplecil/cifar224_5_5_vit-b_simplecil.json \
  --wandb \
  --wandb_entity <username_cua_ban> \
  --wandb_project RevisitingCIL-Test \
  --wandb_name test_simplecil_cifar224
```

---

## 6. Cách xem kết quả Table 1 trên giao diện W&B

Mỗi run sau khi chạy xong sẽ tự động đẩy đầy đủ các trường sau lên W&B:

1. **Summary metrics (để lập bảng như Table 1 trong bài báo):**
   - `avg_acc` (và `table1/avg_acc`): Độ chính xác trung bình qua toàn bộ các task ($\bar{\mathcal{A}}$).
   - `last_acc` (và `table1/last_acc`): Độ chính xác ở task cuối cùng ($\mathcal{A}_B$).
   - `top5_avg_acc`, `top5_last_acc`: Phiên bản Top-5 tương ứng.
   - `total_params`, `trainable_params`: Tổng tham số và số tham số được tinh chỉnh.
2. **Biểu đồ theo từng Task (để vẽ đường cong như Figure 4):**
   - `eval/top1`: Độ chính xác Top-1 sau mỗi task ($\mathcal{A}_b$).
   - `eval/top5`, `eval/old_acc`, `eval/new_acc`, `eval/avg_acc_so_far`.
   - `task/task_id`, `task/seen_classes` (dùng làm trục X *Number of Classes*).
3. **Biểu đồ quá trình Adapt ở Task 0 (với các phương pháp có train):**
   - `adapt/epoch`, `adapt/loss`, `adapt/train_acc`, `adapt/lr`.

**Mẹo xem dạng bảng giống Table 1 trên W&B:**
Mở project trên W&B $\rightarrow$ tab **Runs (Table)** $\rightarrow$ chọn **Group by $\rightarrow$ `group`** (tức `<dataset>_init<...>_inc<...>`) $\rightarrow$ bật thêm cột `method_name`, `avg_acc`, `last_acc` $\rightarrow$ toàn bộ 6 phương pháp trên từng bộ dữ liệu sẽ xếp cạnh nhau trực quan.

---

### 6. Chạy Exp 2A & 2B (W&B)

- **Exp 2A (SimpleCIL‑Adapted)**:
  ```bash
  !python main.py --config ./exps/exp2_submodules/imageneta_100_5_simplecil_adapted.json --wandb --tuned_epoch 2
  ```
- **Exp 2A (APER‑Finetune)**:
  ```bash
  !python main.py --config ./exps/exp2_submodules/imageneta_100_5_aper_finetune.json --wandb --tuned_epoch 2
  ```
- **Exp 2B (cifar224‑10‑adapt‑stages)** (nếu có):
  ```bash
  !python main.py --config ./exps/exp2_submodules/cifar224_10_10_vit-b_adapter.json --wandb --tuned_epoch 2
  ```

Lưu ý: Tham số `--tuned_epoch` giúp chạy nhanh để kiểm tra; bỏ qua để dùng giá trị mặc định (`20`). Các file JSON đã được cập nhật với `backbone_lr`, `warmup_epochs`, `grad_clip` để tối ưu quá trình fine‑tune.

---

# 7. Xử lý một số lỗi thường gặp

- **`403 Forbidden` hoặc không tạo được run trên `nnm-hcmus/Group7-ML-Lab`:**
  Kiểm tra tài khoản W&B của bạn đã chấp nhận lời mời vào team `nnm-hcmus` chưa, hoặc kiểm tra xem bạn đã dán đúng API key của đúng tài khoản đó chưa. Trong lúc chờ cấp quyền, có thể truyền `--wandb_entity <username_cua_ban> --wandb_project <ten_project>` để log tạm về workspace cá nhân.
- **`SecretNotFoundError: Secret WANDB_API_KEY does not exist` trên Colab:**
  Kiểm tra lại tên secret phải viết hoa chính xác `WANDB_API_KEY` và công tắc **Notebook access** đã được bật sang màu xanh.
- **Colab ngắt kết nối giữa chừng:**
  Lần chạy mới sẽ tạo một run mới trên W&B; các dữ liệu đã tải vào `/content/RevisitingCIL_ML/data` trong cùng runtime Colab vẫn giữ nguyên (trừ khi runtime bị reset hoàn toàn).
