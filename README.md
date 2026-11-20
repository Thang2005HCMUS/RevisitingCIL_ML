# RevisitingCIL_ML

Đồ án tái lập và mở rộng các thực nghiệm từ bài báo **"Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need"** (Da-Wei Zhou et al., IJCV 2024) trên nền tảng PyTorch và Weights & Biases (`nnm-hcmus/Group7-ML-Lab`).

Thay vì chỉ chạy lại mã nguồn gốc từ [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT), nhóm xây dựng lại toàn bộ pipeline cho `SimpleCIL` và khung `APER` theo hướng tách bạch rõ từng thành phần (*Data*, *Backbone*, *Tuning Strategy*, *Dual-Branch Encoder*, *Cosine Prototype Classifier*), đồng bộ với hệ sinh thái `timm >= 1.0` trên Google Colab, đồng thời chạy đối chiếu song song với mã nguồn gốc để kiểm chứng từng cơ chế cốt lõi của bài báo.

---

## Một số ghi nhận trong quá trình tái tạo

Trong quá trình viết lại và đối chiếu với bài báo, nhóm ghi nhận vài điểm đáng chú ý về mặt kỹ thuật lẫn lý thuyết:

- **Tính suy biến của APER về SimpleCIL:** Trong thiết kế hợp nhất hai nhánh `DualBranchEncoder`, ngay cả khi nhánh thích nghi không được cập nhật (trường hợp `SimpleCIL`), việc nối đôi vector $[\phi(x), \phi(x)]$ lên $1536$ chiều vẫn cho giá trị Cosine Similarity đồng nhất với không gian $768$ chiều gốc do tính bất biến tỉ lệ: $\cos([\phi(x), \phi(x)], [p_i, p_i]) = \cos(\phi(x), p_i)$. Điều này xác nhận trực tiếp nhận định ở cuối Mục 4.1 rằng `SimpleCIL` là cận dưới tự nhiên của `APER`.
- **Khác biệt môi trường `timm`:** Code gốc phụ thuộc chặt vào `timm==0.6.12` (đặc biệt ở các đường dẫn nội bộ cho `PatchEmbed`, `DropPath` và `pretrained_cfg['url']` trong SSF/Adapter). Khi chuyển lên `timm >= 1.0`, nhóm đã viết lại cơ chế nạp trọng số chuẩn từ `timm.create_model` kết hợp lọc `state_dict` và quản lý rõ việc mở khóa tham số PET (`adaptmlp`, `ssf_scale`/`ssf_shift_`, `Prompt_Tokens`) thay vì để ngầm định trong lớp khởi tạo backbone.
- **Độ nhạy của bước Adapt ở Task 0:** Cơ chế chỉ thích nghi một lần ở chặng đầu tiên giúp APER tránh quên thảm khốc và không làm lệch hệ quy chiếu giữa nguyên mẫu các lớp cũ và mới, nhưng cũng đặt ra câu hỏi thực tế về tỷ lệ ghép cứng $1:1$ khi độ lệch miền (domain gap) thay đổi — đây chính là động cơ cho Thực nghiệm 2 và 3 bên dưới.

---

## Cấu trúc dự án

```text
RevisitingCIL_ML/
├── main.py                  # Điểm khởi chạy CLI (--config, --wandb, --merge_mode, --adapt_stages)
├── exps/                    # Bộ file cấu hình JSON cho Table 1, ResNet và Thực nghiệm 2
├── src/
│   ├── data/                # Đọc 6 dataset và chia luồng nhiệm vụ CIL (ContinualTaskManager)
│   ├── convs/               # Định nghĩa ViT tích hợp Adapter, SSF, VPT (tương thích timm >= 1.0)
│   ├── models/              # Khởi tạo PTM, DualBranchEncoder và CosinePrototypeClassifier
│   ├── tuning/              # Các chiến lược tinh chỉnh (Freeze, Finetune, VPT, SSF, Adapter, BN)
│   ├── framework/           # APERLearner, Evaluator và tích hợp ghi log W&B
│   └── utils/               # Đếm tham số và thiết lập file log
├── tests/                   # Unit tests kiểm tra luồng model, merge_mode, adapt_stages và W&B
├── structure.md             # Mô tả chi tiết cây thư mục, hàm quan trọng và luồng thực thi
└── wandb.md                 # Hướng dẫn tạo tài khoản W&B, gắn API key trên Colab và chạy thí nghiệm
```

- Chi tiết kiến trúc và danh sách hàm chính: xem [structure.md](structure.md).
- Hướng dẫn thiết lập môi trường Colab, W&B API key và các câu lệnh chạy mẫu: xem [wandb.md](wandb.md).

---

## Mục tiêu các thực nghiệm

### Thực nghiệm 1 — Tái lập bảng kết quả chính (Table 1)
Đánh giá 6 phương pháp (`SimpleCIL`, `APER w/ Finetune`, `APER w/ VPT-Shallow`, `APER w/ VPT-Deep`, `APER w/ SSF`, `APER w/ Adapter`) dùng chung backbone `ViT-B/16-IN21K` trên 7 bộ dữ liệu chuẩn (`CIFAR-100 B0 Inc5`, `CUB-200 B0 Inc10`, `ImageNet-R B0 Inc5`, `ImageNet-A B0 Inc10`, `ObjectNet B0 Inc10`, `OmniBenchmark B0 Inc30`, `VTAB B0 Inc10`). Nhóm chạy song song cả mã nguồn tự tái tạo và mã nguồn `LAMDA-PILOT` trên cùng một project W&B để đối chiếu hai chỉ số chính: độ chính xác trung bình qua các chặng ($\bar{\mathcal{A}}$) và độ chính xác ở chặng cuối ($\mathcal{A}_B$).

### Thực nghiệm 2 — Giải phẫu cơ chế của APER (Figure 5d & 5f)
Gồm hai nhánh kiểm chứng bản chất hoạt động của APER (xem chi tiết thiết kế tại [Experiment_2_plan.md](Experiment_2_plan.md)):
- **Nhánh 2A (Sub-module Ablation trên `ImageNet-A B100 Inc5`):** So sánh `SimpleCIL-PTM` ($768$d), `SimpleCIL-Adapted` ($768$d), `SimpleCIL-21K+1K` ($1536$d) và `APER w/ Finetune` ($1536$d) để kiểm chứng mức tăng độ chính xác đến từ sự bổ trợ giữa *Generalizability* và *Adaptivity*, không đơn thuần do tăng gấp đôi số chiều đặc trưng.
- **Nhánh 2B (Adapting Stages trên `CIFAR-224 B0 Inc10`):** Khảo sát `APER w/ Adapter` khi cho phép thích nghi qua $T \in \{0, 1, \dots, 10\}$ chặng đầu tiên trong khi đóng băng trọng số phân loại của các lớp cũ, qua đó quan sát trực tiếp hiện tượng dịch chuyển không gian đặc trưng (feature drift) gây mất tương thích nguyên mẫu khi $T \ge 2$.

### Thực nghiệm 3 (Bổ sung) — Khảo sát độ nhạy hai chiều $(\alpha, k)$
Thực nghiệm mở rộng do nhóm tự thiết kế (chi tiết tại [Experiment_3_plan.md](Experiment_3_plan.md)), mở rộng phép ghép cứng $1:1$ của APER thành pha trộn có trọng số $\Phi_\alpha(x) = [\sqrt{\alpha}\hat{\phi}^*(x), \sqrt{1-\alpha}\hat{\phi}(x)]$ với $\alpha \in [0, 1]$, kết hợp nén chiều bằng PCA xuống $k \in \{16, 32, \dots, 1536\}$ (đối chứng với Random Subsampling). Mục tiêu là khảo sát sự dịch chuyển của $\alpha^*$ theo mức độ domain gap (`CIFAR-100` vs `ImageNet-R` / `CUB-200`) và tìm đường biên Pareto giữa độ chính xác và dung lượng lưu trữ nguyên mẫu.

---

## Phụ lục: Experiment Mapping Table

> Bảng ánh xạ giữa từng thực nghiệm, script/notebook tạo ra kết quả, link W&B của từng run theo 7 bộ dữ liệu và nhận xét ngắn về điều run đó cho thấy.  
> *(Ghi chú định dạng ô dataset: điền `[W&B](<url>) (A_bar / A_B)` hoặc `N/A` nếu không chạy).*

### Bảng A — Thực nghiệm 1: So sánh mã nguồn tái tạo (Ours) và PILOT (Table 1, ViT-B/16-IN21K)

| Phương pháp | Phiên bản | Script / Notebook | CIFAR-100 (`B0 Inc5`) | CUB-200 (`B0 Inc10`) | ImageNet-R (`B0 Inc5`) | ImageNet-A (`B0 Inc10`) | ObjectNet (`B0 Inc10`) | OmniBench (`B0 Inc30`) | VTAB (`B0 Inc10`) | Nhận xét chính (What the run shows) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SimpleCIL** | Ours | `main.py` (`exps/simplecil/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | — | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **SimpleCIL** | PILOT | `PILOT` (`exps/simplecil.json`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ Finetune** | Ours | `main.py` (`exps/aper_finetune/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | — | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ Finetune** | PILOT | `PILOT` (`exps/aper_finetune.json`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ VPT-Shallow** | Ours | `main.py` (`exps/aper_vpt-shallow/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | — | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ VPT-Shallow** | PILOT | `PILOT` (`exps/aper_vpt.json`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ VPT-Deep** | Ours | `main.py` (`exps/aper_vpt-deep/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | — | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ VPT-Deep** | PILOT | `PILOT` (`exps/aper_vpt.json`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ SSF** | Ours | `main.py` (`exps/aper_ssf/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | — | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ SSF** | PILOT | `PILOT` (`exps/aper_ssf.json`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ Adapter** | Ours | `main.py` (`exps/aper_adapter/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | — | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |
| **APER w/ Adapter** | PILOT | `PILOT` (`exps/aper_adapter.json`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *(Điền nhận xét ngắn)* |

### Bảng B — Thực nghiệm 2 & 3 (Ablation & Phân tích mở rộng)

| Thực nghiệm | Cấu hình / Biến thể | Script / Notebook | Link W&B | Nhận xét chính (What the run shows) |
| :--- | :--- | :--- | :---: | :--- |
| **Exp 2A** | `SimpleCIL-PTM` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_simplecil_ptm.json` | [W&B]() | *(Điền nhận xét ngắn)* |
| **Exp 2A** | `SimpleCIL-Adapted` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_simplecil_adapted.json` | [W&B]() | *(Điền nhận xét ngắn)* |
| **Exp 2A** | `SimpleCIL-21K+1K` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_simplecil_21k_1k.json` | [W&B]() | *(Điền nhận xét ngắn)* |
| **Exp 2A** | `APER w/ Finetune` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_aper_finetune.json` | [W&B]() | *(Điền nhận xét ngắn)* |
| **Exp 2B** | `Adapt-0` $\rightarrow$ `Adapt-10` (`CIFAR-224 B0 Inc10`) | `main.py --config exps/exp2_adapt_stages/cifar224_10_10_vit-b_adapter.json --adapt_stages T` | [W&B Group]() | *(Điền nhận xét ngắn)* |
| **Exp 3** | 2D Sweep $(\alpha, k)$ (`CIFAR-224` & `ImageNet-R` / `CUB-200`) | *(Điền script/notebook thực nghiệm 3)* | [W&B]() | *(Điền nhận xét ngắn)* |
