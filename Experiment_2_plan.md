# BÁO CÁO TỔNG QUAN THỰC NGHIỆM 2 (EXPERIMENT 2)
**Chủ đề:** Giải phẫu Cơ chế Hợp nhất Đặc trưng (Sub-module Ablation) và Khảo sát Động học Thích nghi Đa chặng (Adapting Stages Analysis)  
**Tham chiếu bài báo gốc:** Mục 5.3.2 (Figure 5d) & Mục 5.3.4 (Figure 5f) — *Zhou et al., IJCV 2024*  
**Dự án:** `RevisitingCIL_ML` (W&B Project: `nnm-hcmus/Group7-ML-Lab`)

---

## 1. Đặt vấn đề & Mục tiêu Khoa học

Sau khi Thực nghiệm 1 (Table 1) chứng minh hiệu năng vượt trội của khung làm việc **APER (Adapt and mERge)** trên 7 bộ dữ liệu tiêu chuẩn, Thực nghiệm 2 đóng vai trò là trụ cột kiểm chứng **bản chất cơ học** đằng sau sự thành công của APER. Cụ thể, thực nghiệm này được thiết kế để trả lời hai câu hỏi nghiên cứu cốt lõi (Research Questions - RQs):

* **RQ1 (Kiểm chứng tính hợp lực Generalizability – Adaptivity vs. Số chiều đặc trưng):** Khi APER ghép nối vector đặc trưng của mô hình đã tinh chỉnh $\phi^*(x) \in \mathbb{R}^d$ và mô hình tiền huấn luyện gốc $\phi(x) \in \mathbb{R}^d$ thành $[\phi^*(x), \phi(x)] \in \mathbb{R}^{2d}$, độ chính xác tăng lên là nhờ sự bổ trợ thực sự giữa *tính thích nghi (Adaptivity)* và *tính khái quát (Generalizability)*, hay chỉ đơn thuần nhờ việc tăng gấp đôi số chiều không gian biểu diễn từ $d = 768$ lên $2d = 1536$?
* **RQ2 (Kiểm chứng chiến lược chỉ thích nghi tại Stage đầu tiên):** Tại sao APER lại giới hạn việc tinh chỉnh mô hình duy nhất ở chặng đầu tiên ($\mathcal{D}^1$, tức $T=1$) rồi đóng băng hoàn toàn ở các chặng sau? Điều gì sẽ xảy ra với không gian nguyên mẫu (prototype space) nếu mô hình tiếp tục được tinh chỉnh tuần tự ở $T$ chặng tiếp theo ($T \in \{2, 3, \dots, B\}$)?

---

## 2. Thiết kế Thực nghiệm & Cơ sở Toán học

Thực nghiệm 2 được chia thành hai nhánh độc lập nhưng bổ trợ chặt chẽ cho nhau, bám sát thiết lập chuẩn của bài báo gốc.

### 2.1. Nhánh 2A: Giải phẫu các Thành phần Cấu thành (Compositional Components — Tái lập Figure 5d)

* **Tập dữ liệu & Phân chia CIL:** `ImageNet-A` với cấu hình `B100 Inc5` (Stage đầu tiên gồm $m = 100$ lớp cơ sở, 20 stage tiếp theo mỗi stage thêm $n = 5$ lớp mới; tổng cộng $B = 21$ stages trên 200 lớp). Đây là tập dữ liệu có độ lệch miền (domain gap) lớn so với ImageNet gốc, giúp làm nổi bật vai trò của bước Adapt.
* **Kiến trúc Backbone:** `ViT-B/16-IN21K` ($d = 768$).
* **Chiến lược tinh chỉnh ở Stage 1:** `Fully Finetune` (tối ưu hóa toàn bộ tham số mạng trên $\mathcal{D}^1$ theo Công thức 7 trong 20 epochs).
* **4 Cấu hình Mô hình Đối sánh:**

| Tên cấu hình | Hàm trích xuất đặc trưng $\Phi(x)$ | Số chiều | Ý nghĩa đánh giá |
| :--- | :--- | :---: | :--- |
| **1. `SimpleCIL-PTM`** | $\phi_{\text{IN21K}}(x)$ | $768$ | Chỉ sử dụng *Generalizability* thuần túy của PTM đóng băng. |
| **2. `SimpleCIL-Adapted`** | $\phi^*(x)$ | $768$ | Chỉ sử dụng *Adaptivity* thuần túy của mô hình sau khi finetune trên $\mathcal{D}^1$. |
| **3. `SimpleCIL-21K+1K`** | $[\phi_{\text{IN21K}}(x), \phi_{\text{IN1K}}(x)]$ | $1536$ | Ghép 2 mô hình PTM đóng băng (`IN21K` và `IN1K`). Có cùng số chiều $1536$ như APER nhưng không thích nghi với miền đích. |
| **4. `APER w/ Finetune`** | $[\phi^*(x), \phi_{\text{IN21K}}(x)]$ | $1536$ | Hợp nhất cả *Adaptivity* ($\phi^*$) và *Generalizability* ($\phi_{\text{IN21K}}$). |

### 2.2. Nhánh 2B: Khảo sát Số lượng Chặng Thích nghi (Influence of Adapting Stages — Tái lập Figure 5f)

* **Tập dữ liệu & Phân chia CIL:** `CIFAR-100` (`cifar224`) với cấu hình `B0 Inc10` (10 stages, mỗi stage học thêm 10 lớp mới: $\mathcal{D}^1, \mathcal{D}^2, \dots, \mathcal{D}^{10}$).
* **Kiến trúc Backbone & PEFT:** `ViT-B/16-IN21K` kết hợp với `APER w/ Adapter` (AdaptMLP với chiều chiếu giảm $r = 16$).
* **Quy trình Huấn luyện Tổng quát theo ngưỡng chặng $T \in \{0, 1, 2, \dots, 10\}$:**
  * Gọi $T$ là số lượng chặng đầu tiên mà mô hình được phép cập nhật tham số Adapter $\Theta = \theta_{W_{\text{down}}} \cup \theta_{W_{\text{up}}} \cup \theta_W$ (ký hiệu trên biểu đồ là `Adapt-T`).
  * **Khi $T = 0$ (`Adapt-0`):** Không tinh chỉnh ở bất kỳ chặng nào; mô hình suy biến về `SimpleCIL`.
  * **Khi $T = 1$ (`Adapt-1`):** Chính là thuật toán `APER w/ Adapter` tiêu chuẩn (chỉ tinh chỉnh ở Stage 1 trên $\mathcal{D}^1$).
  * **Khi $T \ge 2$ (`Adapt-2` $\rightarrow$ `Adapt-10`):**
    1. Trong các chặng $b \in \{1, 2, \dots, T\}$: Mở khóa tham số Adapter trên nhánh $\phi^*(\cdot)$, đóng băng các vector trọng số phân loại (prototypes) của các lớp cũ $\mathcal{Y}_{b-1}$ để hạn chế quên trực tiếp tại tầng phân loại, và huấn luyện mô hình trên tập dữ liệu mới $\mathcal{D}^b$. Kết thúc việc huấn luyện ở chặng $b$, trích xuất prototype cho các lớp mới $i \in Y_b$ bằng đặc trưng ghép $[\phi^{*(b)}(x), \phi(x)]$.
    2. Từ chặng $b > T$ trở đi: Đóng băng vĩnh viễn cả hai nhánh $[\phi^{*(T)}(\cdot), \phi(\cdot)]$ và chỉ trích xuất prototype cho các lớp mới theo đúng Công thức (6).

---

## 3. Kế hoạch Tích hợp vào Kiến trúc Mã nguồn (`RevisitingCIL_ML`)

Nhờ thiết kế hướng đối tượng sẵn có trong thư mục `src/`, toàn bộ Thực nghiệm 2 được cài đặt bằng cách mở rộng nhẹ 2 module cốt lõi mà không làm thay đổi luồng chạy của Table 1:

### 3.1. Mở rộng `DualBranchEncoder` (`src/models/merged_encoder.py`)
Bổ sung tham số cấu hình `merge_mode` (mặc định là `"both"` để tương thích ngược với Table 1):
* `merge_mode == "both"`: Trả về `torch.cat([feat_adapted, feat_ptm], dim=-1)` ($1536$ chiều — dùng cho `APER`).
* `merge_mode == "adapted_only"`: Chỉ chạy forward qua `adapted_ptm` và trả về `feat_adapted` ($768$ chiều — dùng cho `SimpleCIL-Adapted`).
* `merge_mode == "ptm_only"`: Chỉ chạy forward qua `frozen_ptm` và trả về `feat_ptm` ($768$ chiều — dùng cho `SimpleCIL-PTM`).
* `merge_mode == "dual_ptm"`: Nạp `ViT-B/16-IN21K` vào nhánh 1 và `ViT-B/16-IN1K` vào nhánh 2 (cả hai cùng đóng băng), trả về `torch.cat([feat_21k, feat_1k], dim=-1)` ($1536$ chiều — dùng cho `SimpleCIL-21K+1K`).

### 3.2. Mở rộng `APERLearner` (`src/framework/learner.py`)
* Đọc thêm siêu tham số `adapt_stages` $T$ từ file JSON cấu hình (mặc định `adapt_stages = 1`).
* Trong hàm `execute_task(self, task_id, train_loader, proto_loader, test_loader)`:
  * Thay điều kiện kiểm tra chặng đầu tiên `if task_id == 0:` bằng `if task_id < self.args.get("adapt_stages", 1):`.
  * Khi $1 \le \text{task\_id} < T$: Giữ nguyên các vector nguyên mẫu cũ $W_{\text{old}} = [w_1, \dots, w_{|\mathcal{Y}_{b-1}|}]$ (đóng băng `requires_grad = False`), khởi tạo thêm nhánh phân loại cho các lớp thuộc chặng hiện tại $Y_b$, cập nhật trọng số Adapter của `adapted_ptm` bằng SGD trong 20 epochs, sau đó trích xuất prototype của các lớp trong $Y_b$ ghép vào `CosinePrototypeClassifier`.

### 3.3. Tổ chức File Cấu hình Thực nghiệm (`exps/`)
* `exps/exp2_submodules/`: Chứa 4 file JSON cho `imageneta` (`B100 Inc5`) ứng với 4 giá trị `merge_mode`.
* `exps/exp2_adapt_stages/`: Chứa 11 file JSON (hoặc 1 script truyền cờ `--adapt_stages T` qua CLI) cho `cifar224` (`B0 Inc10`) với $T \in \{0, 1, \dots, 10\}$.

---

## 4. Chỉ số Đánh giá & Kết quả Kỳ vọng

Toàn bộ quá trình chạy được ghi log tự động lên Weights & Biases (`nnm-hcmus/Group7-ML-Lab`) theo từng stage $b \in \{1, \dots, B\}$ để trích xuất hai biểu đồ trung tâm:

1. **Biểu đồ Đường cong Hiệu năng Nhánh 2A (Tái lập Figure 5d):**
   * *Kỳ vọng:* Đường `APER w/ Finetune` nằm cao nhất xuyên suốt 21 stages, vượt xa `SimpleCIL-21K+1K` ($\sim +8\%$ đến $+10\%$) dù cả hai có cùng số chiều đặc trưng $1536$. Điều này bác bỏ hoàn toàn giả thuyết APER mạnh hơn chỉ nhờ gấp đôi kích thước vector.
   * *Hiện tượng trôi đặc trưng:* `SimpleCIL-Adapted` xuất phát ở Stage 1 rất cao (xấp xỉ `APER`), nhưng tốc độ suy giảm độ chính xác ở các stage sau nhanh hơn so với `APER`, chứng minh rằng việc finetune đơn thuần làm ghi đè các đặc trưng mức cao (overwriting high-level features) và cần nhánh PTM gốc để neo giữ tính khái quát.
2. **Biểu đồ Đường cong Hiệu năng Nhánh 2B (Tái lập Figure 5f):**
   * *Kỳ vọng:* Đường `Adapt-1` ($T=1$) giữ vị trí cao nhất và ổn định nhất khi số lớp tăng tới 100.
   * *Sự sụp đổ khi $T \ge 2$:* Với các cấu hình `Adapt-2` đến `Adapt-10`, ngay tại thời điểm mô hình tiếp tục tinh chỉnh ở Stage 2, 3,..., không gian biểu diễn của $\phi^{*(b)}(\cdot)$ bị dịch chuyển (feature drift). Hệ quả là các prototype của lớp cũ (được trích xuất từ $\phi^{*(1)}$) trở nên **không tương thích (incompatible)** với các prototype của lớp mới (được trích xuất từ $\phi^{*(b)}$), làm độ chính xác sụt giảm mạnh (có thể tụt xuống thấp hơn cả `Adapt-0` / `SimpleCIL`).