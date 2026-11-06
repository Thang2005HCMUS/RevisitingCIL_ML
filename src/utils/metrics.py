import torch

def count_parameters(model: torch.nn.Module, trainable: bool = False) -> int:
    """
    Đếm số lượng tham số của mô hình[cite: 20].
    Nếu trainable=True, chỉ đếm các tham số requires_grad=True.
    """
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor2numpy(x: torch.Tensor):
    """
    Chuyển đổi tensor sang numpy array an toàn (tự động xử lý CPU/GPU và đồ thị tính toán)[cite: 20].
    """
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def target2onehot(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Chuyển đổi nhãn (labels) sang dạng One-Hot Encoding[cite: 20].
    """
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot