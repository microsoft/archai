# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py

import os
import random
from contextlib import contextmanager
from typing import Generator, Optional, Union

import numpy as np
import torch


def init_distributed(use_cuda: bool) -> None:
    """Initialize distributed backend for parallel training.

    This method sets up the distributed backend for parallel training based on the specified
    `use_cuda` flag. If `use_cuda` is `True`, it initializes the distributed mode using the
    CUDA/NCCL backend. Otherwise, it uses the Gloo backend.

    Args:
        use_cuda (bool): Whether to initialize the distributed mode using the CUDA/NCCL backend.

    Raises:
        AssertionError: If the distributed mode is not initialized successfully.

    """

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    distributed = world_size > 1
    if distributed:
        backend = "nccl" if use_cuda else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        assert torch.distributed.is_initialized()


def barrier() -> None:
    """Synchronize all processes in the distributed backend.

    This method calls the `torch.distributed.barrier` function if the distributed mode is
    available and initialized. The `barrier` function synchronizes all processes in the
    distributed backend by blocking the processes until all processes have reached this point.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank() -> int:
    """Return the rank of the current process in the distributed backend.

    Returns:
        The rank of the current process in the distributed backend. If the distributed mode
            is not available or not initialized, the returned rank will be `0`.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    return 0


def get_world_size() -> int:
    """Return the total number of processes in the distributed backend.

    Returns:
        The total number of processes in the distributed backend. If the distributed mode
            is not available or not initialized, the returned world size will be `1`.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    return 1


def all_reduce(tensor: Union[int, float, torch.Tensor], op: Optional[str] = "sum") -> Union[int, float]:
    """Reduce the input tensor/value into a scalar using the specified reduction operator.

    This method applies the specified reduction operator to the input tensor/value in a distributed
    manner. The result is a scalar value that is computed by aggregating the values from all
    processes in the distributed backend.

    Args:
        tensor: Input tensor/value to be reduced.
        op: Type of reduction operator. The supported operators are "sum", "mean",
            "min", "max", and "product".

    Returns:
        The scalar value obtained by applying the reduction operator to the input
            tensor/value. If the distributed mode is not available or not initialized,
            the inputvtensor/value is returned as is.

    Raises:
        RuntimeError: If the specified reduction operator is not supported.

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
    """Context manager for synchronizing the processes in the distributed backend.

    This context manager yields the rank of the current process in the distributed backend and
    synchronizes all processes on exit.

    Yields:
        The rank of the current process in the distributed backend.

    Example:
        >>> with sync_workers():
        >>>     # Execute some code that should be synchronized across all processes.
        >>>     pass

    """

    rank = get_rank()
    yield rank

    barrier()


def get_cuda_device_name() -> str:
    """Get the name of the CUDA devices.

    Returns:
        Name of the CUDA devices.

    """

    return ", ".join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def init_cuda(seed: int, local_rank: Optional[int] = 0) -> None:
    """Setup CUDA for distributed training.

    Args:
        seed: The seed to use for initializing the random number generator.
        local_rank: The local rank of the current process in the distributed backend.

    """

    seed = seed + local_rank

    # Set the seed for generating random numbers
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set CUDNN-related options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Set to false if deterministic
    torch.set_printoptions(precision=10)
