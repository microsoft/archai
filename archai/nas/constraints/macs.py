from typing import Tuple

import tensorwatch as tw
import torch


def calculate_macs(model: torch.nn.Module, input_shape: Tuple) -> int:
    """Computes MACs for an architecture.

    Args:
        arch (torch.nn.Module): Model architecture.
        x (torch.Tensor): Sample input.

    Returns:
        int: MACs of the architecture.
    """    
    model_stats = tw.ModelStats(model, input_shape, clone_model=True)
    return model_stats.MAdd
