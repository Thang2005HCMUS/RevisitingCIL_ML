import logging
import torch
import numpy as np

def evaluate(encoder, classifier, test_loader, nb_old: int, device: torch.device, increment: int = 10, topk: int = 5):
    """
    Hàm đánh giá mô hình trên tập test, trích xuất logits và thu thập nhãn dự đoán[cite: 5].
    """
    encoder.eval()
    classifier.eval()
    
    y_pred, y_true = [], []
    
    with torch.no_grad():
        for _, inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            # Trích xuất đặc trưng từ [phi*(x), phi(x)] và tính Cosine Logits[cite: 6]
            features = encoder(inputs)
            logits = classifier(features)
            
            # Lấy top-k dự đoán lớn nhất[cite: 6]
            predicts = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[1] 
            
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())
            
    # Gộp toàn bộ kết quả của các batch[cite: 6]
    y_pred = np.concatenate(y_pred)  # Kích thước: [N, topk]
    y_true = np.concatenate(y_true)  # Kích thước: [N]
    
    return _compute_metrics(y_pred, y_true, nb_old, increment, topk)


def _compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, nb_old: int, increment: int, topk: int):
    """
    Tính toán các chỉ số độ đo chi tiết cho CIL: Top-1, Top-5, In-Task, Old-Task[cite: 5, 7].
    """
    assert len(y_pred) == len(y_true)
    
    metrics = {}
    
    # Độ chính xác Top-1 trên toàn bộ dữ liệu (Total Accuracy)[cite: 7]
    metrics["top1"] = np.around((y_pred[:, 0] == y_true).sum() * 100 / len(y_true), decimals=2)
    
    # Độ chính xác Top-5[cite: 5]
    metrics[f"top{topk}"] = np.around(
        (y_pred == np.tile(y_true[:, None], (1, topk))).sum() * 100 / len(y_true), 
        decimals=2
    )

    # Độ chính xác nhóm (Grouped Accuracy) chia theo từng task (increment)[cite: 7]
    grouped = {"total": metrics["top1"]}
    for class_id in range(0, np.max(y_true) + 1, increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        if len(idxes) > 0:
            label = f"{str(class_id).rjust(2, '0')}-{str(class_id + increment - 1).rjust(2, '0')}"
            grouped[label] = np.around((y_pred[idxes, 0] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    metrics["grouped"] = grouped

    # Độ chính xác trên các lớp cũ (Old Classes Accuracy)[cite: 7]
    old_idxes = np.where(y_true < nb_old)[0]
    metrics["old"] = 0.0 if len(old_idxes) == 0 else np.around(
        (y_pred[old_idxes, 0] == y_true[old_idxes]).sum() * 100 / len(old_idxes), decimals=2
    )

    # Độ chính xác trên các lớp mới (New Classes Accuracy)[cite: 7]
    new_idxes = np.where(y_true >= nb_old)[0]
    metrics["new"] = 0.0 if len(new_idxes) == 0 else np.around(
        (y_pred[new_idxes, 0] == y_true[new_idxes]).sum() * 100 / len(new_idxes), decimals=2
    )

    return metrics