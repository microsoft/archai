# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Parallel data utilities to help distributed settings.
"""

from typing import Any, Dict, List, Optional
import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply


def scatter(inputs: torch.Tensor,
            target_gpus: List[int],
            chunk_sizes: List[int],
            dim: Optional[int] = 0) -> None:
    """Slices tensors into approximately equal chunks and
        distributes them across given GPUs.
    
    Note that it also duplicates references to objects that are not tensors.

    Args:
        inputs: Input tensor.
        target_gpus: Available GPUs.
        chunk_sizes: Size of sliced tensors.
        dim: Dimension to slice tensors.

    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()

        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))

        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))

        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))

        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs: torch.Tensor,
                   kwargs: Dict[str, Any],
                   target_gpus: List[int],
                   chunk_sizes: List[int],
                   dim: Optional[int] = 0) -> None:
    """Slices tensors into approximately equal chunks and
        distributes them across given GPUs (with kwargs support).
    
    Note that it also duplicates references to objects that are not tensors.

    Args:
        inputs: Input tensor.
        target_gpus: Available GPUs.
        chunk_sizes: Size of sliced tensors.
        dim: Dimension to slice tensors.

    """

    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []

    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

    inputs = tuple(inputs)
    kwargs = tuple(kwargs)

    return inputs, kwargs


class BalancedDataParallel(DataParallel):
    """Provides a DataParallel class that balances the data between devices.

    """

    def __init__(self, gpu0_bsz: int, *args, **kwargs) -> None:
        """Overrides initialization method.

        Args:
            gpu0_bsz: Batch size of GPU #0.

        """

        self.gpu0_bsz = gpu0_bsz

        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        """Performs forward pass over the class.

        Returns:
            (torch.Tensor): Scattered output tensors.

        """

        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids

        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids)

        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)

        return self.gather(outputs, self.output_device)

    def parallel_apply(self,
                       replicas: List[int],
                       device_ids: List[int],
                       inputs: torch.Tensor,
                       kwargs: Dict[str, Any]) -> callable:
        """Applies parallel distribution.

        Args:
            replicas: Replicas identifiers.
            device_ids: List of device identifiers.
            inputs: Input tensor.
            kwargs: Additional keyword arguments.

        Returns:
            (callable): Distributed parallel function.

        """

        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self,
                inputs: torch.Tensor,
                kwargs: Dict[str, Any],
                device_ids: List[int]) -> callable:
        """Performs the scattering procedure.

        Args:
            inputs: Input tensor.
            kwargs: Additional keyword arguments.
            device_ids: List of devices identifiers.

        Returns:
            (callable): Actual scatter function.

        """

        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)

        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)

            for i in range(delta):
                chunk_sizes[i + 1] += 1

            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]

        else:
            return super().scatter(inputs, kwargs, device_ids)

        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
