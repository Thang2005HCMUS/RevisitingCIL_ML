import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# ==============================================================================
# 1. LỚP BỌC DỮ LIỆU DÀNH CHO PYTORCH DATALOADER
# ==============================================================================

class ContinualDataset(Dataset):
    """
    Lớp Wrapper chuẩn hóa dataset cho PyTorch DataLoader.
    Thay thế cho lớp 'DummyDataset' trong code gốc của tác giả.
  [cite: 7]
    """
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Số lượng ảnh và nhãn không khớp!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            # Dữ liệu dạng đường dẫn file (ImageNet, CUB, v.v.)
            with open(self.images[idx], "rb") as f:
                image = Image.open(f).convert("RGB")
        else:
            # Dữ liệu dạng mảng numpy array (CIFAR)
            image = Image.fromarray(self.images[idx])
        
        # Trsf luôn luôn là transforms.Compose hoàn chỉnh
        image = self.trsf(image)
        label = self.labels[idx]

        return idx, image, label

# ==============================================================================
# 2. HÀM XỬ LÝ TRANSFORMS (CẮT, LẬT CƠ BẢN)
# ==============================================================================

def build_transform(is_train: bool, input_size: int = 224):
    """
    Tạo bộ Augmentation chuẩn cho các dataset kích thước lớn như ImageNet-R, CUB.
  [cite: 8]
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.05, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        size = int((256 / 224) * input_size)
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])

def split_images_labels(imgs):
    """Tách đường dẫn ảnh và nhãn từ torchvision.datasets.ImageFolder[cite: 6]."""
    images = [item[0] for item in imgs]
    labels = [item[1] for item in imgs]
    return np.array(images), np.array(labels)

# ==============================================================================
# 3. QUẢN LÝ DỮ LIỆU CÁC DATASET CỤ THỂ
# ==============================================================================

class BaseData:
    """Class cơ sở cho mọi bộ dữ liệu."""
    use_path = False
    train_trsf = None
    test_trsf = None
    class_order = []
    
    def download_data(self):
        raise NotImplementedError

class CIFAR100Data(BaseData):
    use_path = False
    class_order = np.arange(100).tolist()
    
    # Định nghĩa cứng transform cho ảnh nhỏ (32x32) của CIFAR[cite: 8]
    train_trsf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    
    test_trsf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    def download_data(self):
        train_dataset = datasets.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.CIFAR100("./data", train=False, download=True)
        
        self.train_data = train_dataset.data
        self.train_targets = np.array(train_dataset.targets)
        
        self.test_data = test_dataset.data
        self.test_targets = np.array(test_dataset.targets)

class ImageNetRData(BaseData):
    use_path = True
    class_order = np.arange(200).tolist()
    
    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)

    def download_data(self):
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"
        
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class CUBData(BaseData):
    use_path = True
    class_order = np.arange(200).tolist()
    
    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)

    def download_data(self):
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"
        
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

# ==============================================================================
# 4. FACTORY GET_DATASET
# ==============================================================================

def get_dataset(dataset_name: str) -> BaseData:
    """Trả về instance của Dataset tương ứng."""
    name = dataset_name.lower()
    
    if name == "cifar100":
        return CIFAR100Data()
    elif name == "imagenetr":
        return ImageNetRData()
    elif name == "cub":
        return CUBData()
    # Bạn có thể bổ sung ObjectNet, OmniBenchmark, ImageNet-A tại đây nếu cần[cite: 8].
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' chưa được hỗ trợ.")