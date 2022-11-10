# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

"""Utilities for initializing distributed training.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional, Union

import torch


def init_distributed(use_cuda: bool) -> None:
    """Initializes distributed backend.

    Args: 
        use_cuda: Whether to initialize distributed mode using the CUDA/NCLL backend, e.g.,
            `False` will use Gloo.

    """

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    distributed = (world_size > 1)
    if distributed:
        backend = "nccl" if use_cuda else "gloo"
        torch.distributed.init_process_group(backend=backend,
                                             init_method="env://")

        assert torch.distributed.is_initialized()


def barrier() -> None:
    """Calls torch.distributed.barrier() if using distributed mode.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank() -> int:
    """Gets the distributed rank.

    Returns:
        (int): Distributed rank, where `0` indicates that distributed mode was
            not initialized.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    return 0


def get_world_size() -> int:
    """Gets the total number of distributed workers.

    Returns:
        (int): Total number of distributed workers, where `1` indicates that distributed
            mode was not initialized.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    return 1
    

def all_reduce(tensor: Union[int, float, torch.Tensor], op: Optional[str] = "sum") -> Union[int, float]:
    """Reduces tensor into a scalar value when using distributed mode.

    Args:
        tensor: Input tensor/value.
        op: Type of reduction operator.

    Returns:
        (Union[int, float]): Scalar value.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if op == "sum" or op == "mean":
            torch_op = torch.distributed.ReduceOp.SUM
        elif op == "min":
            torch_op = torch.distributed.ReduceOp.MIN
        elif op == "max":
            torch_op = torch.distributed.ReduceOp.MAX
        elif op == "product":
            torch_op = torch.distributed.ReduceOp.PRODUCT
        else:
            raise RuntimeError(f"Operator: {op} is not supported yet.")

        backend = torch.distributed.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device("cuda")
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device("cpu")
        else:
            raise RuntimeError(f"Distributed backend: {backend} is not supported yet.")

        tensor = torch.tensor(tensor, device=device)
        torch.distributed.all_reduce(tensor, torch_op)
        if op == "mean":
            tensor /= get_world_size()

        return tensor.item()
    
    return tensor


@contextmanager
def sync_workers() -> Generator[int, None, None]:
    """Yields the distributed rank and synchronizes all workers on exit.
    
    """

    rank = get_rank()
    yield rank

    barrier()
