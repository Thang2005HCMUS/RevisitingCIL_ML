

### 1. Cấu trúc thư mục dự án (Project Layout)

```text
revisiting_cil/
│
├── configs/                          # Quản lý cấu hình thử nghiệm (YAML/JSON)
│   ├── simplecil_cifar100.yaml
│   └── aper_adapter_imagenetr.yaml
│
├── src/                              # Mã nguồn chính
│   ├── data/                         # Quản lý dữ liệu & phân chia CIL tasks
│   │   ├── __init__.py
│   │   ├── datasets.py               # Dataset wrapper (CIFAR, ImageNet-R, CUB...)[cite: 8, 20]
│   │   └── task_manager.py           # Chia luồng dữ liệu theo từng Task/Session[cite: 4, 20]
│   │
│   ├── models/                       # Trích xuất đặc trưng & Bộ phân loại
│   │   ├── __init__.py
│   │   ├── backbones.py              # Load mô hình tiền huấn luyện từ timm/torchvision[cite: 20]
│   │   ├── classifiers.py            # Cosine classifier & Prototype manager[cite: 6, 7]
│   │   └── merged_encoder.py         # Module hợp nhất [phi*(x), phi(x)][cite: 6, 7]
│   │
│   ├── tuning/                       # Thiết kế theo Strategy Pattern cho PET
│   │   ├── __init__.py
│   │   ├── base.py                   # Giao diện TuningStrategy trừu tượng
│   │   ├── adapter.py                # Kỹ thuật Adapter/AdaptFormer[cite: 4, 7]
│   │   ├── prompt.py                 # Visual Prompt Tuning (VPT Shallow/Deep)[cite: 4, 7]
│   │   ├── scale_shift.py            # Scale & Shift (SSF)[cite: 4, 7]
│   │   └── linear_probe.py           # Freeze mode cho SimpleCIL[cite: 2, 5]
│   │
│   ├── framework/                    # Luồng điều khiển huấn luyện & đánh giá
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Đánh giá Accuracy theo từng chặng (Average/Last)[cite: 8, 9, 20]
│   │   └── learner.py                # Incremental Learner điều phối APER / SimpleCIL[cite: 7, 20]
│   │
│   └── utils/                        # Công cụ hỗ trợ
│       ├── __init__.py
│       ├── logger.py                 # Ghi log tiến trình thực nghiệm[cite: 22]
│       └── metrics.py                # Các hàm phụ trợ tính toán (Cosine distance, Acc)[cite: 6, 20]
│
├── experiments/                      # Scripts chạy thử nghiệm và lưu log kết quả
├── tests/                            # Unit tests kiểm tra tính đúng đắn của code
├── README.md
├── requirements.txt
└── main.py                           # Điểm khởi chạy chương trình (Entry point)[cite: 21]

```

---

### 2. Bảng phân rã kiến trúc OOP (Design Pattern & Class Responsibilities)

| Thành phần (Component) | Class chính | Pattern áp dụng | Nhiệm vụ & Trách nhiệm (Single Responsibility) | Tính linh hoạt / Mở rộng |
|---|---|---|---|---|
| **Data Orchestration** | `ContinualTaskManager` | *Iterator Pattern* | Đọc dataset, xáo trộn class order, chia dữ liệu thành $B$ task (base classes và incremental classes). | Đổi tập dữ liệu (CIFAR-100, CUB, ImageNet-R...) không ảnh hưởng logic model. |
| **Adaptation Mechanism** | `TuningStrategy` (Interface)<br>• `AdapterStrategy`<br>• `PromptStrategy`<br>• `SSFStrategy`<br>• `FreezeStrategy` | *Strategy Pattern* | Chịu trách nhiệm cấy tham số học nhẹ (lightweight parameters) hoặc cấu hình cờ `requires_grad` lên PTM tại Stage 1. | Muốn thử nghiệm kỹ thuật PET mới (như LoRA, BitFit) chỉ cần viết thêm 1 class Strategy mới. |
| **Feature Representation** | `DualBranchEncoder` | *Decorator / Composite* | Nhận vào mô hình PTM gốc $\phi(\cdot)$ và mô hình đã thích ứng $\phi^*(\cdot)$, nối đặc trưng thành $[\phi^*(x), \phi(x)]$. | Dễ dàng chèn thêm module giảm chiều (PCA layer/Random projection) theo bài báo. |
| **Classifier Head** | `CosinePrototypeClassifier` | *State / Observer Pattern* | Lưu trữ trọng số nguyên mẫu $w_c = p_c$, tự động mở rộng ma trận phân loại khi xuất hiện lớp mới và tính Cosine Logits. | Tách biệt hoàn toàn việc tính prototype ra khỏi luồng huấn luyện backbone. |
| **Core Workflow** | `APERLearner` | *Template Method Pattern* | Điều phối toàn bộ vòng đời CIL: Task 1 (Adapt -> Merge -> Prototypes), Task $2 \dots B$ (Chỉ trích xuất Prototypes). | Tái sử dụng một bộ khung duy nhất cho cả SimpleCIL lẫn mọi biến thể APER. |

### 3. Thiết kế chi tiết giao diện lớp (Class Interfaces Blueprint)

* **`TuningStrategy` (Trừu tượng hóa cách tinh chỉnh):**
```python
class TuningStrategy(ABC):
    @abstractmethod
    def prepare_adapted_model(self, backbone: nn.Module) -> nn.Module:
        """Cấy module PET hoặc unfreeze các tham số cần tối ưu."""
        pass

```


* **`CosinePrototypeClassifier` (Quản lý Prototype & Cosine Loss):**
```python
class CosinePrototypeClassifier(nn.Module):
    def __init__(self, in_features: int, scale: float = 16.0): ...
    def register_prototypes(self, new_prototypes: torch.Tensor):
        """Nối thêm class center trung bình từ embedding của task mới.""" ...
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Tính normalized cosine similarity: (w / ||w||) * (f / ||f||).""" ...

```


* **`DualBranchEncoder` (Ghép đặc trưng linh hoạt):**
```python
class DualBranchEncoder(nn.Module):
    def __init__(self, frozen_ptm: nn.Module, adapted_ptm: nn.Module): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nối đặc trưng: [phi*(x), phi(x)]
        return torch.cat([self.adapted_ptm(x), self.frozen_ptm(x)], dim=-1)

```


* **`APERLearner` (Điều phối tổng thể):**
```python
class APERLearner:
    def __init__(self, backbone: nn.Module, strategy: TuningStrategy, ...): ...
    def execute_task(self, task_id: int, train_loader: DataLoader, class_ids: List[int]):
        if task_id == 0:
            # Giai đoạn Adapt và Merge chỉ chạy 1 lần ở Task 1
            self._adapt_and_merge(train_loader, class_ids)
        # Tính trung bình prototype cho các lớp của task hiện tại
        self._update_prototypes(train_loader, class_ids)

```



---

### 4. Luồng dữ liệu và tương tác giữa các Class (Workflow Execution)

```text
[DataLoader (Task 1)] 
         │
         ▼
[TuningStrategy] ─── (Gắn Adapter/SSF/Prompt) ───► [Adapted Model phi*]
                                                            │
  ┌────────────────── [Đóng băng cả 2 nhánh] ────────────────┘
  │
  ▼
[DualBranchEncoder: [phi*(x), phi(x)]]
         │
         ├───► Trích xuất toàn bộ feature của Task b
         │
         ▼
[Tính Prototype: p_c = Mean(Features_c)]
         │
         ▼
[CosinePrototypeClassifier] ─── (Cập nhật vector weights w_c) ───► [Dự đoán Test Set]

```

