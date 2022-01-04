# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data distribution helpers and utilities.
"""

import os
from contextlib import contextmanager
from typing import Optional

import torch


def init_distributed(cuda: bool) -> bool:
    """Initializes the distributed backend.

    Args:
        cuda: Whether to initialize the nccl backend (True) or gloo backend (False).

    Returns:
        (bool): Boolean defining which backend has been initialized.

    """

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = (world_size > 1)

    if distributed:
        backend = 'nccl' if cuda else 'gloo'

        torch.distributed.init_process_group(backend=backend,
                                             init_method='env://')

        assert torch.distributed.is_initialized()

    return distributed


def barrier() -> None:
    """Calls torch.distributed.barrier() if distributed is in use.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank() -> int:
    """Gets the distributed rank or returns zero if distributed is not initialized.

    Returns:
        (int): Distributed rank.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    return rank


def get_world_size() -> int:
    """Gets the total number of distributed workers or returns one if distributed is
        not initialized.

    Returns:
        (int): Size of the distributed workers.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    return world_size


def all_reduce_item(value: torch.Tensor, op: Optional[str] = 'sum') -> torch.Tensor:
    """All-reduces single scalar value if distributed is in use.

    Args:
        value: Value to be reduced.
        op: Operator to be used in reduction.

    Returns:
        (torch.Tensor): Reduced tensor.

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if op == 'sum' or op == 'mean':
            dop = torch.distributed.ReduceOp.SUM
        elif op == 'min':
            dop = torch.distributed.ReduceOp.MIN
        elif op == 'max':
            dop = torch.distributed.ReduceOp.MAX
        elif op == 'product':
            dop = torch.distributed.ReduceOp.PRODUCT
        else:
            raise RuntimeError('Unsupported reduce op')

        backend = torch.distributed.get_backend()

        if backend == torch.distributed.Backend.NCCL:
            device = torch.device('cuda')
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device('cpu')
        else:
            raise RuntimeError('Unsupported distributed backend')

        tensor = torch.tensor(value, device=device)
        torch.distributed.all_reduce(tensor, dop)

        if op == 'mean':
            tensor /= get_world_size()

        ret = tensor.item()
    else:
        ret = value

    return ret


@contextmanager
def sync_workers() -> int:
    """Yields distributed rank and synchronizes all workers on exit.

    Yields:
        (int): Distributed rank.
        
    """

    rank = get_rank()

    yield rank

    barrier()
