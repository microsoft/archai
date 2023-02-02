from typing import Optional, List, Dict
import torch

@torch.no_grad()
def get_confusion_matrix(pred_labels: torch.LongTensor,
                         true_labels: torch.LongTensor,
                         num_labels: int, ignore_index: int = 255) -> torch.LongTensor:
    pred_labels, true_labels = pred_labels.view(-1), true_labels.view(-1)
    
    ignore_mask = (true_labels == ignore_index)
    pred_labels, true_labels = pred_labels[~ignore_mask], true_labels[~ignore_mask]

    confusion_matrix = num_labels * true_labels + pred_labels
    return torch.bincount(confusion_matrix, minlength=num_labels**2).reshape(num_labels, num_labels)


@torch.no_grad()
def get_iou(confusion_matrix: torch.LongTensor,
            ignore_labels: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    ignore_labels = ignore_labels or []
    ignore_labels = torch.isin(
        torch.arange(len(confusion_matrix)), torch.tensor(ignore_labels)
    )
    
    diag = confusion_matrix.diag()
    row_sum = confusion_matrix.sum(dim=1)
    col_sum = confusion_matrix.sum(dim=0)
    
    class_iou = (diag + 1e-7) / (row_sum + col_sum - diag + 1e-7)

    return {
        'class_iou': class_iou,
        'mIOU': class_iou[~ignore_labels].mean(),
    }


@torch.no_grad()
def get_f1_scores(confusion_matrix: torch.LongTensor,
                  ignore_labels: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    ignore_labels = ignore_labels or []
    ignore_labels = torch.isin(
        torch.arange(len(confusion_matrix)), torch.tensor(ignore_labels)
    )

    recall = confusion_matrix.diag() / (confusion_matrix.sum(dim=1) + 1e-7)
    prec = confusion_matrix.diag() / (confusion_matrix.sum(dim=0) + 1e-7)

    class_f1 = 2 * prec * recall / (prec + recall + 1e-7)
    support = confusion_matrix.sum(dim=1)[~ignore_labels]

    return {
        'class_f1': class_f1,
        'macro_f1': class_f1[~ignore_labels].mean(),
        'weighted_f1': (class_f1[~ignore_labels] * support).sum() / support.sum()
    }
