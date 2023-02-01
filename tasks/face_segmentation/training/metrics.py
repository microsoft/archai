from typing import Optional, List
import torch

@torch.no_grad()
def get_confusion_matrix(pred_labels: torch.LongTensor, true_labels: torch.LongTensor,
                         num_labels: int, ignore_index: int = 255) -> torch.LongTensor:
    pred_labels, true_labels = pred_labels.view(-1), true_labels.view(-1)
    
    ignore_mask = (true_labels == ignore_index)
    pred_labels, true_labels = pred_labels[~ignore_mask], true_labels[~ignore_mask]

    confusion_matrix = num_labels * true_labels + pred_labels
    return torch.bincount(confusion_matrix, minlength=num_labels**2).reshape(num_labels, num_labels)


@torch.no_grad()
def get_iou(confusion_matrix: torch.LongTensor, ignore_labels: Optional[List[int]] = None) -> torch.Tensor:
    ignore_labels = ignore_labels or []
    
    diag = confusion_matrix.diag()
    row_sum = confusion_matrix.sum(dim=1)
    col_sum = confusion_matrix.sum(dim=0)
    
    class_iou = (diag + 1e-7) / (row_sum + col_sum - diag + 1e-7)
    class_iou[ignore_labels] = torch.nan
    
    return class_iou
