# BÁO CÁO TỔNG QUAN THỰC NGHIỆM 3 (ADDITIONAL ANALYSIS)
**Chủ đề:** Khảo sát Độ nhạy Hai chiều: Tỷ lệ Pha trộn Đặc trưng Có trọng số ($\alpha$-Mixing) và Nén Chiều Không gian Biểu diễn bằng PCA ($k$-Compression) qua các Miền Dữ liệu  
**Phân loại:** Thực nghiệm Mở rộng Tự thiết kế (Team-Designed Sensitivity Sweep — Core Scope)  
**Cơ sở mở rộng từ bài báo:** Mở rộng Công thức (6) (Mục 4.1) kết hợp Mục 5.3.1 (Figure 5a–5c)  
**Dự án:** `RevisitingCIL_ML` (W&B Project: `nnm-hcmus/Group7-ML-Lab`)

---

## 1. Động cơ Khoa học & Đóng góp của Nhóm

Theo yêu cầu của **Core Scope**, bên cạnh việc tái lập các thực nghiệm chính của bài báo, nhóm tự thiết kế một phân tích chuyên sâu nhằm khảo sát các giới hạn chưa được khai thác trong công trình gốc của Zhou et al. (IJCV 2024). Khi phân tích cơ chế hợp nhất đặc trưng (Merge) của APER tại Công thức (6), nhóm nhận thấy **hai hạn chế cố hữu**:

1. **Giả định Tỷ lệ Ghép nối Cứng nhắc (Rigid 1:1 Concatenation):**
   APER nối trực tiếp $[\phi^*(x), \phi(x)]$ với trọng số đóng góp ngang nhau ($50\%$ *Adaptivity* : $50\%$ *Generalizability*). Tuy nhiên, mức độ lệch miền (domain gap) giữa tập dữ liệu tiền huấn luyện (ImageNet-21K) và tập dữ liệu đích là rất khác nhau:
   * Trên các tập dữ liệu gần với ImageNet (như `CIFAR-100`), bản thân PTM gốc đã có tính khái quát rất cao; việc gán trọng số quá lớn cho $\phi^*(x)$ (vốn chỉ được học trên một tập con lớp nhỏ $\mathcal{D}^1$ gồm 5–10 lớp) có nguy cơ gây hiện tượng quá khớp cục bộ vào Task 0 (Task-0 bias).
   * Ngược lại, trên các tập dữ liệu có sự dịch chuyển phân phối mạnh hoặc đòi hỏi đặc trưng chuyên biệt (như tranh vẽ/phác thảo trong `ImageNet-R` hoặc phân loại chim chi tiết trong `CUB-200`), nhánh $\phi^*(x)$ đóng vai trò quyết định và cần được ưu tiên trọng số cao hơn.
2. **Chi phí Bộ nhớ và Tính toán Gấp đôi (Doubled Feature Dimensionality):**
   Việc ghép hai vector làm tăng gấp đôi kích thước mỗi prototype từ $d = 768$ lên $2d = 1536$. Trong các kịch bản học tăng cường dài hạn với hàng nghìn lớp hoặc triển khai trên thiết bị biên (edge devices), việc lưu trữ và tính tích vô hướng Cosine trên không gian 1536 chiều chứa nhiều chiều dư thừa (redundant dimensions).

**Mục tiêu của Experiment 3:** Thực hiện một khảo sát độ nhạy hai chiều (**2D Sensitivity Sweep**) trên không gian siêu tham số $(\alpha, k)$ — trong đó $\alpha \in [0, 1]$ điều khiển tỷ lệ pha trộn giữa *Adaptivity* và *Generalizability*, còn $k \in \{16, \dots, 1536\}$ là số chiều sau khi nén bằng Phân tích Thành phần Chính (PCA) — nhằm tìm ra đường biên Pareto tối ưu giữa **Độ chính xác ($\bar{\mathcal{A}}, \mathcal{A}_B$)** và **Chi phí lưu trữ ($k$)** trên các mức độ Domain Gap khác nhau.

---

## 2. Cơ sở Toán học của Phương pháp Mở rộng

### 2.1. Hợp nhất Đặc trưng Có trọng số ($\alpha$-Weighted Feature Mixing)

Để kiểm soát chính xác tỷ lệ đóng góp của từng nhánh vào bộ phân loại Cosine mà không bị ảnh hưởng bởi sự chênh lệch chuẩn độ dài (norm scale) giữa hai mạng, trước tiên ta chuẩn hóa $\ell_2$ từng vector đặc trưng độc lập:
$$\hat{\phi}^*(x) = \frac{\phi^*(x)}{\|\phi^*(x)\|_2} \in \mathbb{R}^d, \qquad \hat{\phi}(x) = \frac{\phi(x)}{\|\phi(x)\|_2} \in \mathbb{R}^d$$

Với hệ số pha trộn $\alpha \in [0, 1]$, vector đặc trưng hợp nhất có trọng số $\Phi_\alpha(x) \in \mathbb{R}^{2d}$ được định nghĩa là:
$$\Phi_\alpha(x) = \left[ \sqrt{\alpha} \, \hat{\phi}^*(x), \; \sqrt{1 - \alpha} \, \hat{\phi}(x) \right]$$

**Tính chất toán học quan trọng:**  
Với mọi mẫu $x, x'$, chuẩn của vector hợp nhất luôn bằng đơn vị ($\|\Phi_\alpha(x)\|_2^2 = \alpha \|\hat{\phi}^*(x)\|_2^2 + (1-\alpha)\|\hat{\phi}(x)\|_2^2 = 1$), và độ tương đồng Cosine giữa hai mẫu bất kỳ phân rã tuyến tính hoàn hảo theo $\alpha$:
$$\text{CosineSim}\big(\Phi_\alpha(x), \Phi_\alpha(x')\big) = \alpha \, \big\langle \hat{\phi}^*(x), \hat{\phi}^*(x') \big\rangle + (1 - \alpha) \, \big\langle \hat{\phi}(x), \hat{\phi}(x') \big\rangle$$

* **Khi $\alpha = 0.0$:** Mô hình bỏ qua hoàn toàn nhánh thích nghi, suy biến về `SimpleCIL-PTM` ($100\%$ Generalizability).
* **Khi $\alpha = 0.5$:** Khôi phục chính xác tỷ lệ đóng góp $1:1$ của `APER` gốc.
* **Khi $\alpha = 1.0$:** Mô hình bỏ qua hoàn toàn nhánh PTM gốc, suy biến về `SimpleCIL-Adapted` ($100\%$ Adaptivity).

### 2.2. Nén Chiều Không gian Đặc trưng bằng PCA ($k$-Dimensional Projection)

Để tuân thủ quy định chỉ sử dụng **Standard Scientific Stack** (`scikit-learn`), ngay sau khi hoàn tất việc tinh chỉnh ở Stage 1 trên $\mathcal{D}^1$, ta trích xuất tập hợp các vector đặc trưng $\mathcal{Z}^1 = \{\Phi_\alpha(x_j)\}_{j=1}^{|\mathcal{D}^1|}$ và khớp (fit) một phép chiếu trực giao PCA giảm từ $2d = 1536$ chiều xuống $k$ chiều ($k \le \min(2d, |\mathcal{D}^1|)$):
$$\tilde{\Phi}_{\alpha, k}(x) = W_{\text{PCA}}^{(k)\top} \big(\Phi_\alpha(x) - \mu_{\mathcal{D}^1}\big) \in \mathbb{R}^k$$
Trong đó $\mu_{\mathcal{D}^1} \in \mathbb{R}^{2d}$ là vector trung bình mẫu trên $\mathcal{D}^1$ và $W_{\text{PCA}}^{(k)} \in \mathbb{R}^{2d \times k}$ chứa $k$ vector riêng ứng với $k$ trị riêng lớn nhất của ma trận hiệp phương sai trên $\mathcal{D}^1$.

Ở tất cả các chặng $b \in \{1, 2, \dots, B\}$, ma trận chiếu $W_{\text{PCA}}^{(k)}$ được giữ cố định. Nguyên mẫu của lớp $i \in Y_b$ được tính trực tiếp trên không gian nén $k$ chiều:
$$p_i^{(\alpha, k)} = \frac{1}{K} \sum_{j=1}^{|\mathcal{D}^b|} \mathbb{I}(y_j = i) \, \tilde{\Phi}_{\alpha, k}(x_j) \in \mathbb{R}^k$$
Dự đoán cuối cùng được thực hiện thông qua `CosinePrototypeClassifier` trên không gian $\mathbb{R}^k$. Bên cạnh PCA, nhóm cài đặt thêm phương pháp đối chứng **Random Subsampling** (lấy ngẫu nhiên $k$ chiều từ $1536$ chiều gốc với 3 random seeds) để làm rõ hiệu quả bảo toàn thông tin của PCA.

---

## 3. Thiết kế Ma trận Thực nghiệm & Tối ưu Hóa Thời gian Chạy

### 3.1. Không gian Lưới Tham số (Hyperparameter Sweep Grid)
* **Trục 1 — Hệ số pha trộn $\alpha$ (11 mức):**
  $$\alpha \in \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$$
* **Trục 2 — Số chiều nén $k$ (8 mức, theo cấp số nhân cơ số 2):**
  $$k \in \{16, 32, 64, 128, 256, 512, 768, 1536\}$$
  *(Lưu ý: Tại $k = 1536$, ta giữ nguyên không gian ghép nối mà không cắt giảm chiều để làm mốc baseline).*
* **Tổng số cấu hình đánh giá trên mỗi bộ dữ liệu:** $11 \times 8 = 88$ cấu hình CIL trọn vẹn (tương đương 88 lần chạy qua toàn bộ $B$ tasks).

### 3.2. Lựa chọn Tập dữ liệu Đối chiếu (Cross-Domain Gap Comparison)
Để chứng minh tính thích ứng của siêu tham số $(\alpha, k)$ theo đặc thù dữ liệu, nhóm thực hiện khảo sát trên **2 bộ dữ liệu đại diện cho 2 thái cực Domain Gap** (sử dụng chung backbone `ViT-B/16-IN21K` và phương pháp `APER w/ Adapter` hoặc `APER w/ SSF`):
1. **Miền gần với dữ liệu gốc (Small Domain Gap):** `CIFAR-100` (`cifar224`, cấu hình `B0 Inc5`, 20 tasks).
2. **Miền lệch phân phối mạnh / Chuyên biệt (Large Domain Gap / OOD):** `ImageNet-R` (`imagenetr`, cấu hình `B0 Inc5`, 40 tasks) và/hoặc `CUB-200` (`cub`, cấu hình `B0 Inc10`, 20 tasks).

### 3.3. Kỹ thuật Tăng tốc "Zero-Retraining Offline Feature Caching" (Chạy 88 cấu hình trong < 2 phút)
Một ưu điểm kỹ thuật cực lớn của thiết kế này là cả $\alpha$ và phép chiếu PCA $k$ đều tác động **sau khi hai nhánh $\phi^*(x)$ và $\phi(x)$ đã được đóng băng hoàn toàn ở cuối Stage 1**:
1. **Bước 1 (Chạy trên GPU — 1 lần duy nhất cho mỗi dataset, mất ~3–5 phút):** Huấn luyện nhánh `adapted_ptm` ($\phi^*$) ở Task 0 trong 20 epochs. Sau đó, chạy forward pass một lượt qua toàn bộ các tập dữ liệu train/test và lưu đệm (cache) bộ đôi tensor $(\phi^*(x), \phi(x)) \in \mathbb{R}^{N \times 768}$ vào bộ nhớ RAM/đĩa.
2. **Bước 2 (Quét lưới 88 cấu hình trên CPU/GPU — mất < 90 giây):** Vòng lặp lồng nhau qua 11 giá trị $\alpha$ và 8 giá trị $k$ chỉ thực hiện phép toán đại số tuyến tính trên các vector đã cache sẵn (`L2-norm` $\rightarrow$ nhân $\sqrt{\alpha}, \sqrt{1-\alpha}$ $\rightarrow$ `sklearn.decomposition.PCA(n_components=k)` $\rightarrow$ tính trung bình prototype $\rightarrow$ nhân ma trận Cosine Similarity).

---

## 4. Đặc tả Tích hợp vào Codebase (`RevisitingCIL_ML`)

Không cần sửa đổi các file huấn luyện hiện tại, nhóm bổ sung một module phân tích gọn nhẹ vào dự án:

| Thành phần / File | Chi tiết cài đặt |
| :--- | :--- |
| `src/models/merged_encoder.py` | Bổ sung thuộc tính `alpha: float = 0.5` và `projector: Optional[Callable] = None` vào `DualBranchEncoder`. Hỗ trợ chuẩn hóa từng nhánh trước khi nối nếu bật cờ `weighted_norm=True`. |
| `src/analysis/sensitivity_sweep.py` *(File mới)* | Chứa hàm `extract_and_cache_features(learner, task_manager)` và `run_2d_sensitivity_sweep(cached_features, alpha_grid, k_grid, method="pca")`. Tự động tính $\bar{\mathcal{A}}$ (`avg_acc`), $\mathcal{A}_B$ (`last_acc`), độ chính xác trên lớp cũ (`old_acc`) và lớp mới (`new_acc`) cho từng cặp $(\alpha, k)$. |
| `exps/exp3_analysis/` *(Thư mục mới)* | Chứa file cấu hình JSON `sweep_cifar224_adapter.json` và `sweep_imagenetr_adapter.json` quy định lưới `alpha_grid` và `k_grid`. |
| **W&B Logging & Visualization** | Tự động đẩy bảng tổng hợp `wandb.Table` (88 dòng) và vẽ 3 biểu đồ chất lượng cao (`alpha_curve.png`, `pca_compression_curve.png`, `heatmap_alpha_k.png`) lưu vào `experiments/exp3_analysis/`. |

---

## 5. Các Biểu đồ & Phát hiện Kỳ vọng (Đưa vào Slide Thuyết trình)

Thực nghiệm 3 cung cấp 3 biểu đồ phân tích sâu sắc đóng vai trò điểm nhấn học thuật riêng của nhóm:

1. **Biểu đồ 1 — Đường cong Đánh đổi $\alpha$-Mixing ($\bar{\mathcal{A}}$ và $\mathcal{A}_B$ theo $\alpha \in [0, 1]$ ở $k=1536$):**
   * Làm rõ đường cong chữ U ngược (inverted U-curve): Tại hai đầu mút $\alpha = 0$ (chỉ dùng PTM) và $\alpha = 1$ (chỉ dùng Adapted), hiệu năng đều thấp hơn phần giữa, khẳng định sự cần thiết của cả hai yếu tố.
   * **Phát hiện mới so với bài báo:** Đỉnh tối ưu $\alpha^*$ dịch chuyển phụ thuộc vào Domain Gap! Trên `CIFAR-100`, đỉnh thường lệch về phía $\alpha^* \in [0.3, 0.4]$ (ưu tiên *Generalizability* của PTM để tránh thiên lệch vào 5 lớp đầu tiên); trong khi trên `ImageNet-R` / `CUB-200`, đỉnh dịch chuyển về phía $\alpha^* \in [0.6, 0.7]$ (ưu tiên *Adaptivity* để thích ứng với đặc trưng miền mới).
2. **Biểu đồ 2 — Hiệu quả Nén chiều: PCA Projection vs. Random Subsampling theo $k \in [16, 1536]$:**
   * Chứng minh đường cong PCA hội tụ cực nhanh: Chỉ với **$k = 64$ hoặc $k = 128$ chiều** (giảm **91.6%** dung lượng bộ nhớ lưu trữ classifier so với $1536$ chiều), PCA đã khôi phục được $>99\%$ độ chính xác của mô hình APER đầy đủ và vượt xa `SimpleCIL` ($768$ chiều).
   * Ở vùng chiều cực thấp ($k \in \{16, 32\}$), PCA vượt trội hơn hẳn Random Subsampling từ $15\% - 25\%$ độ chính xác nhờ khả năng gom phương sai chính từ tập dữ liệu $\mathcal{D}^1$.
3. **Biểu đồ 3 — Bản đồ Nhiệt 2D (2D Heatmap của $\bar{\mathcal{A}}$ và $\mathcal{A}_B$ trên lưới $\alpha \times k$):**
   * Trực quan hóa vùng hoạt động ổn định ("Sweet Spot Region"): Chỉ ra rằng ngay cả khi nén đặc trưng xuống $k = 128$ chiều, việc chọn đúng tỷ lệ pha trộn $\alpha \in [0.4, 0.7]$ vẫn giúp mô hình đạt hiệu năng cao hơn cấu hình mặc định ($\alpha = 0.5, k = 1536$) của bài báo gốc.