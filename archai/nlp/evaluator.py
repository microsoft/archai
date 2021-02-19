from typing import List
import numpy as np

import torch
from torch import nn
from torch.utils.data import \
    SubsetRandomSampler, Sampler, Subset, ConcatDataset, Dataset, random_split,\
    DataLoader

from archai.common.apex_utils import ApexUtils


def evaluate(net:nn.Module, data_loader:DataLoader, apex:ApexUtils, device):
    net.eval()

    with torch.no_grad():
        pred_counts = []
        losses:List[torch.Tensor] = []
        prev_seg_state = None

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            preds = net.forward(inputs, labels, prev_seg_state=prev_seg_state)
            prev_seg_state = preds['state']
            losses.append(preds['loss'])
            pred_counts.append(labels.size(0))
        pred_counts = np.array(pred_counts)
        pred_counts_normed = pred_counts / pred_counts.sum()
        loss = sum(x * w for x, w in zip(losses, pred_counts_normed))

    if apex.is_dist():
        gathered_losses = [torch.zeros_like(loss) for _ in range(apex.world_size)]
        torch.distributed.all_gather(gathered_losses, loss)
        loss = sum(gathered_losses) / len(gathered_losses)

    loss = loss.item()
    perplexity = np.nan if loss > 5 else np.e ** loss
    return dict(loss=loss, perplexity=perplexity)