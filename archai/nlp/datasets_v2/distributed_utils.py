# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os

import torch


def start_distributed(use_cuda: bool) -> bool:
    """
    """

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1

    if is_distributed:
        backend = 'nccl' if use_cuda else 'gloo'
        torch.distributed.init_process_group(backend=backend, init_method='env://')
        assert torch.distributed.is_initialized()

    return is_distributed


def get_world_size() -> int:
    """
    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    return world_size


def get_rank() -> int:
    """
    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    return rank
