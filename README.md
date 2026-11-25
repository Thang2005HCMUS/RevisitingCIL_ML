# RevisitingCIL_ML

Đồ án tái lập và mở rộng các thực nghiệm từ bài báo **"Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need"** (Da-Wei Zhou et al., IJCV 2024) trên nền tảng PyTorch và Weights & Biases (`nnm-hcmus/Group7-ML-Lab`).

Thay vì chỉ chạy lại mã nguồn gốc từ [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT), nhóm xây dựng lại toàn bộ pipeline cho `SimpleCIL` và khung `APER` theo hướng tách bạch rõ từng thành phần (*Data*, *Backbone*, *Tuning Strategy*, *Dual-Branch Encoder*, *Cosine Prototype Classifier*), đồng bộ với hệ sinh thái `timm >= 1.0` trên Google Colab, đồng thời chạy đối chiếu song song với mã nguồn gốc để kiểm chứng từng cơ chế cốt lõi của bài báo.

---

## Một số ghi nhận trong quá trình tái tạo

Trong quá trình viết lại và đối chiếu với bài báo, nhóm ghi nhận vài điểm đáng chú ý về mặt kỹ thuật lẫn lý thuyết:

- **Tính suy biến của APER về SimpleCIL:** Trong thiết kế hợp nhất hai nhánh `DualBranchEncoder`, ngay cả khi nhánh thích nghi không được cập nhật (trường hợp `SimpleCIL`), việc nối đôi vector $[\phi(x), \phi(x)]$ lên $1536$ chiều vẫn cho giá trị Cosine Similarity đồng nhất với không gian $768$ chiều gốc do tính bất biến tỉ lệ: $\cos([\phi(x), \phi(x)], [p_i, p_i]) = \cos(\phi(x), p_i)$. Điều này xác nhận trực tiếp nhận định ở cuối Mục 4.1 rằng `SimpleCIL` là cận dưới tự nhiên của `APER`.
- **Khác biệt môi trường `timm`:** Code gốc phụ thuộc chặt vào `timm==0.6.12` (đặc biệt ở các đường dẫn nội bộ cho `PatchEmbed`, `DropPath` và `pretrained_cfg['url']` trong SSF/Adapter). Khi chuyển lên `timm >= 1.0`, nhóm đã viết lại cơ chế nạp trọng số chuẩn từ `timm.create_model` kết hợp lọc `state_dict` và quản lý rõ việc mở khóa tham số PET (`adaptmlp`, `ssf_scale`/`ssf_shift_`, `Prompt_Tokens`) thay vì để ngầm định trong lớp khởi tạo backbone.
- **Độ nhạy của bước Adapt ở Task 0:** Cơ chế chỉ thích nghi một lần ở chặng đầu tiên giúp APER tránh quên thảm khốc và không làm lệch hệ quy chiếu giữa nguyên mẫu các lớp cũ và mới, nhưng cũng đặt ra câu hỏi thực tế về tỷ lệ ghép cứng $1:1$ khi độ lệch miền (domain gap) thay đổi - đây chính là động cơ cho Thực nghiệm 2 và 3 bên dưới.

---

## Cấu trúc dự án

```text
RevisitingCIL_ML/
├── main.py                  # Điểm khởi chạy Exp 1 & 2 (--config, --wandb, --merge_mode, --adapt_stages)
├── run_exp3_sensitivity.py  # Điểm khởi chạy Exp 3 (2D sensitivity sweep alpha x k)
├── exps/                    # Bộ file cấu hình JSON cho Table 1, ResNet, Thực nghiệm 2 & 3
├── src/
│   ├── data/                # Đọc 6 dataset và chia luồng nhiệm vụ CIL (ContinualTaskManager)
│   ├── convs/               # Định nghĩa ViT tích hợp Adapter, SSF, VPT (tương thích timm >= 1.0)
│   ├── models/              # Khởi tạo PTM, DualBranchEncoder và CosinePrototypeClassifier
│   ├── tuning/              # Các chiến lược tinh chỉnh (Freeze, Finetune, VPT, SSF, Adapter, BN)
│   ├── framework/           # APERLearner, Evaluator và tích hợp ghi log W&B
│   ├── analysis/            # Phân tích mở rộng Exp 3 (caching, 2D sweep, PCA, xuất biểu đồ & W&B)
│   └── utils/               # Đếm tham số và chuyển đổi tensor
├── tests/                   # Unit tests kiểm tra luồng model, Exp 2, Exp 3 và W&B
├── structure.md             # Mô tả chi tiết cây thư mục, hàm quan trọng và luồng thực thi
└── wandb.md                 # Hướng dẫn tạo tài khoản W&B, gắn API key trên Colab và chạy thí nghiệm
```

- Chi tiết kiến trúc và danh sách hàm chính: xem [structure.md](structure.md).
- Hướng dẫn thiết lập môi trường Colab, W&B API key và các câu lệnh chạy mẫu: xem [wandb.md](wandb.md).

---

## Mục tiêu các thực nghiệm

### Thực nghiệm 1 - Tái lập bảng kết quả chính (Table 1)
Đánh giá 6 phương pháp (`SimpleCIL`, `APER w/ Finetune`, `APER w/ VPT-Shallow`, `APER w/ VPT-Deep`, `APER w/ SSF`, `APER w/ Adapter`) dùng chung backbone `ViT-B/16-IN21K` trên 7 bộ dữ liệu chuẩn (`CIFAR-100 B0 Inc5`, `CUB-200 B0 Inc10`, `ImageNet-R B0 Inc5`, `ImageNet-A B0 Inc10`, `ObjectNet B0 Inc10`, `OmniBenchmark B0 Inc30`, `VTAB B0 Inc10`). Nhóm chạy song song cả mã nguồn tự tái tạo và mã nguồn `LAMDA-PILOT` trên cùng một project W&B để đối chiếu hai chỉ số chính: độ chính xác trung bình qua các chặng ($\bar{\mathcal{A}}$) và độ chính xác ở chặng cuối ($\mathcal{A}_B$).

### Thực nghiệm 2 - Giải phẫu cơ chế của APER (Figure 5d & 5f)
Gồm hai nhánh kiểm chứng bản chất hoạt động của APER (xem chi tiết thiết kế tại [Experiment_2_plan.md](Experiment_2_plan.md)):
- **Nhánh 2A (Sub-module Ablation trên `ImageNet-A B100 Inc5`):** So sánh `SimpleCIL-PTM` ($768$d), `SimpleCIL-Adapted` ($768$d), `SimpleCIL-21K+1K` ($1536$d) và `APER w/ Finetune` ($1536$d) để kiểm chứng mức tăng độ chính xác đến từ sự bổ trợ giữa *Generalizability* và *Adaptivity*, không đơn thuần do tăng gấp đôi số chiều đặc trưng.
- **Nhánh 2B (Adapting Stages trên `CIFAR-224 B0 Inc10`):** Khảo sát `APER w/ Adapter` khi cho phép thích nghi qua $T \in \{0, 1, \dots, 10\}$ chặng đầu tiên trong khi đóng băng trọng số phân loại của các lớp cũ, qua đó quan sát trực tiếp hiện tượng dịch chuyển không gian đặc trưng (feature drift) gây mất tương thích nguyên mẫu khi $T \ge 2$.

### Thực nghiệm 3 (Bổ sung) - Khảo sát độ nhạy hai chiều $(\alpha, k)$
Thực nghiệm mở rộng do nhóm tự thiết kế (chi tiết tại [Experiment_3_plan.md](Experiment_3_plan.md)), mở rộng phép ghép cứng $1:1$ của APER thành pha trộn có trọng số $\Phi_\alpha(x) = [\sqrt{\alpha}\hat{\phi}^*(x), \sqrt{1-\alpha}\hat{\phi}(x)]$ với $\alpha \in [0, 1]$, kết hợp nén chiều bằng PCA xuống $k \in \{16, 32, \dots, 1536\}$ (đối chứng với Random Subsampling). Mục tiêu là khảo sát sự dịch chuyển của $\alpha^*$ theo mức độ domain gap (`CIFAR-100` vs `ImageNet-R` / `CUB-200`) và tìm đường biên Pareto giữa độ chính xác và dung lượng lưu trữ nguyên mẫu.

---

## Phụ lục: Experiment Mapping Table

Link chạy Ours: [Aper](https://colab.research.google.com/drive/1KZMTSRlqJoxrLuhc6Pdu5FEGVn6lfmCF?usp=sharing)

Link chạy PILOT: [PILOT](https://colab.research.google.com/drive/1eEvqb_cc2BpAjspDmPLxVHFxsg8n6QOC?usp=sharing)

Link chạy Ex2: [Experiment2](https://colab.research.google.com/drive/1VR0FRCH2iUB__DOB2w-MHFG_so8IMJDl?usp=sharing)

Link chạy Ex3: [Experiment3](https://colab.research.google.com/drive/1oR5JFTbkBKOFru87i8xKpLKx6cquWC4E?usp=sharing)

*Các link chạy trên là link mẫu, kết quả chạy thực tế nằm trong project W&B*

> Bảng ánh xạ giữa từng thực nghiệm, script/notebook tạo ra kết quả, link W&B của từng run theo 7 bộ dữ liệu và nhận xét ngắn về điều run đó cho thấy.  
> *(Ghi chú định dạng ô dataset:  `N/A` nếu không chạy).*

### Bảng A — Thực nghiệm 1: So sánh mã nguồn tái tạo (Ours) và PILOT (Table 1, ViT-B/16-IN21K)

| Phương pháp | Phiên bản | Script / Notebook | CIFAR-100 (`B0 Inc5`) | CUB-200 (`B0 Inc10`) | ImageNet-R (`B0 Inc5`) | ImageNet-A (`B0 Inc10`) | ObjectNet (`B0 Inc10`) | OmniBench (`B0 Inc30`) | VTAB (`B0 Inc10`) | Nhận xét chính (What the run shows) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SimpleCIL** | Ours | `main.py` (`exps/simplecil/*`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/6p9v4cd6) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/fzjdz317) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/ztnfi9dd) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/9eim2usa) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/mlsl92lm?nw=nwusernguyennhatminhlk) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/rac5q1pl) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/liqwylo3) | *Trùng khớp gần như tuyệt đối với PILOT và bài báo trên CIFAR (87.57%), CUB (92.23%), IN-R (62.39%), IN-A (60.64%) và OmniBench (79.36%); baseline vững chắc trên đủ 7 bộ dữ liệu (ObjectNet 64.34%).* |
| **SimpleCIL** | PILOT | `PILOT` (`exps/simplecil.json`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/2o8qqenm) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/dfmqwbr0) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/wkynvycx) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/eg6bak61) | N/A | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/bipi4tnh) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/7leopepx) | *Baseline mạnh: CUB đạt 92.23%; vượt APER-Finetune trên 5/6 bộ dữ liệu.* |
| **APER w/ Finetune** | Ours | `main.py` (`exps/aper_finetune/*`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/tdhpb433) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/dn8y9g13) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/p6elz5qr) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/vww2dff0) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/4v4qnv1n?nw=nwusernguyennhatminhlk) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/wbkphcrp) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/bhts1x10) | *Dẫn đầu Avg Acc trên 6/7 bộ dữ liệu (CIFAR 91.25%, CUB 92.36%, IN-R 73.15%, IN-A 63.78%, OmniBench 81.87%, VTAB 88.67%) và Last Acc trên 5/7 bộ nhờ tách backbone_lr/warmup/grad_clip ổn định pha thích nghi.* |
| **APER w/ Finetune** | PILOT | `PILOT` (`exps/aper_finetune.json`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/khb86t80) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/hkip110h) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/fc9j0ipd) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/0zy0neoj) | N/A | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/mx6gxrko) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/biscuigb) | *Thấp hơn SimpleCIL rõ nhất trên CIFAR và OmniBench, lần lượt khoảng 10.46 và 11.09 điểm %.* |
| **APER w/ VPT-Shallow** | Ours | `main.py` (`exps/aper_vpt-shallow/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *Thấp hơn SimpleCIL trên cả 7/7 bộ dữ liệu (giảm mạnh nhất trên ImageNet-A còn 51.88%, -8.76%), cho thấy APER không đảm bảo cận dưới SimpleCIL khi bước thích nghi gây lệch đặc trưng.* |
| **APER w/ VPT-Shallow** | PILOT | `PILOT` (`exps/aper_vpt.json`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/y1f6nj0j) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/3xgvbec8) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/qwak1fqb) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/kowhbnj0) | N/A | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/qd4ngnxv) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/rrtac6tq) | *Vượt VPT-Deep trên CIFAR, ImageNet-R và ImageNet-A; ImageNet-R đạt 64.07%.* |
| **APER w/ VPT-Deep** | Ours | `main.py` (`exps/aper_vpt-deep/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *Dẫn đầu ObjectNet (70.97% / 59.20%, +6.63% so với SimpleCIL) và cải thiện nhẹ trên IN-R (64.78%), nhưng vẫn giảm dưới SimpleCIL trên CIFAR (86.59%) và ImageNet-A (57.44%).* |
| **APER w/ VPT-Deep** | PILOT | `PILOT` (`exps/aper_vpt.json`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/cmsa16yn) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/5jy2zuap) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/d9d12nh4) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/u1vrlims) | N/A | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/97mrm6lj) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/bcbvtdl5) | *Dẫn đầu OmniBench (81.36%) và VTAB (91.41%), nhưng thấp nhất trên ImageNet-A (40.87%).* |
| **APER w/ SSF** | Ours | `main.py` (`exps/aper_ssf/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *Dẫn đầu Last Acc trên ImageNet-R (65.62%; Avg 72.85%, +10.46% so với SimpleCIL) và đứng nhì ObjectNet (70.78%), nhưng giảm mạnh trên ImageNet-A (55.17%, -5.47%).* |
| **APER w/ SSF** | PILOT | `PILOT` (`exps/aper_ssf.json`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/3uo82wot) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/h1bao3xq) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/41f9h0wx) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/cj3lkive) | N/A | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/ekp0k0kl) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/ferbh6ix) | *Dẫn đầu ImageNet-R (67.52%); đứng thứ hai trên CIFAR, OmniBench và VTAB.* |
| **APER w/ Adapter** | Ours | `main.py` (`exps/aper_adapter/*`) | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | [W&B]() | *Ổn định trên CIFAR (89.75%), CUB (92.27%), ObjectNet (68.33%) và OmniBench (80.40%), nhưng chỉ nhích nhẹ trên IN-R (62.96%) và hơi giảm dưới SimpleCIL trên IN-A (60.42%) và VTAB (85.56%).* |
| **APER w/ Adapter** | PILOT | `PILOT` (`exps/aper_adapter.json`) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/owz7k55a) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/by1a9q33) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/y1wyn3k4) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/h0onmdhe) | N/A | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/0os1k85c) | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/zmtr4oks) | *Dẫn đầu 3/6 bộ dữ liệu: CIFAR (90.08%), CUB (92.67%) và ImageNet-A (61.52%).* |

### Bảng B — Thực nghiệm 2 & 3 (Ablation & Phân tích mở rộng)

| Thực nghiệm | Cấu hình / Biến thể | Script / Notebook | Link W&B | Nhận xét chính (What the run shows) |
| :--- | :--- | :--- | :---: | :--- |
| **Exp 2A** | `SimpleCIL-PTM` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_simplecil_ptm.json` | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/nhgl1lrw) | *Baseline PTM đóng băng (768d): giảm từ 57.79% (100 lớp) xuống 48.45% (200 lớp), thấp nhất trong 4 biến thể.* |
| **Exp 2A** | `SimpleCIL-Adapted` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_simplecil_adapted.json` | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/awgctsgb) | *Nhánh thích nghi đơn (768d): đạt cao nhất từ 74.91% đến 63.66% ở chặng cuối (+15.21% so với PTM), cho thấy Adaptivity đóng góp quyết định trên B100.* |
| **Exp 2A** | `SimpleCIL-21K+1K` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_simplecil_21k_1k.json` | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/fhw362a8) | *Ghép hai PTM đóng băng (1536d): đạt 64.68% -> 55.63% (+7.18% so với PTM), cải thiện nhờ đa dạng PTM nhưng vẫn kém rõ các bản có thích nghi.* |
| **Exp 2A** | `APER w/ Finetune` (`IN-A B100 Inc5`) | `main.py --config exps/exp2_submodules/imageneta_100_5_aper_finetune.json` | [W&B](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/zn6bnj78) | *Ghép PTM + Adapted (1536d): đạt 72.41% -> 61.03% (+12.58% so với PTM và +5.40% so với 21K+1K), khẳng định lợi ích vượt xa việc tăng gấp đôi chiều, nhưng vẫn thấp hơn nhánh đơn `SimpleCIL-Adapted` (-2.63%) do trộn cứng tỷ lệ 1:1.* |
| **Exp 2B** | `Adapt-0` $\rightarrow$ `Adapt-10` (`CIFAR-224 B0 Inc10`) | `main.py --config exps/exp2_adapt_stages/cifar224_10_10_vit-b_adapter.json --adapt_stages T` | [Adapt 0](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/awk05y8h) & [Adapt 1](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/4ar9s4ky)| *Adapt-1 cải thiện rõ so với Adapt-0; tiếp tục thích nghi ở T >= 2 gây dịch chuyển đặc trưng (feature drift) làm lệch nguyên mẫu các lớp cũ.* |
| **Exp 3** | 2D Sweep $(\alpha, k)$ (`CIFAR-224` & `ImageNet-R` / `CUB-200`) | `run_exp3_sensitivity.py --config exps/exp3_analysis/sweep_<dataset>_adapter.json` | [IR](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/lxtf7j4y) & [CIFAR](https://forge.coreweave.com/wandb/nnm-hcmus/Group7-ML-Lab/runs/90n1jb5l)| *Cả CIFAR-224 (90.62% vs 89.74%) và ImageNet-R (63.80% vs 63.08%) đều đạt đỉnh ở α*=1.0; PCA vượt Full APER ở k=512 trên CIFAR-224 (90.2%) nhưng bị kém Random Subsampling ở k=32–256 trên ImageNet-R.* |

## Phụ lục: Nhóm Tác giả
### Nhóm 7 - 24TNT1 - Môn Nhập môn học máy - HK1 2026/27 - HCMUS
|MSSV|Họ và tên|Vai trò|
| :--- | :--- | :--- |
| 23122010 | Nguyễn Nhật Minh | 1. Phân tích đề bài, lập kế hoạch, thiết kế kiến trúc và thuật toán. 2. Tìm hiểu cơ bản về các thành phần và tham số trong kiến trúc. 3. Thiết kế thực nghiệm 1,2, và bổ sung. 4. Chạy các thực nghiệm PILOT, thực nghiệm 2 và thực nghiệm 3. |
| 23122011 | Đoàn Hải Nam | 1. Đọc bài báo và nghiên cứu các phương pháp được nêu ra. 2. Trình bày lại trong slide. 3. Chạy một số thực nghiệm. |
| 23122045 | Đoàn Quang Thắng | 1. Cài đặt môi trường phát triển , quản lý thư viện và các tiện ích. 2. Cài đặt lại toàn bộ nội dung SimpleCIL và APER.3. Hỗ trợ Huấn luyện và Thu thập Kết quả. |

