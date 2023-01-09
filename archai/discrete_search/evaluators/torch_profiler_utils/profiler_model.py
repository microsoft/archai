# Copyright (c) DeepSpeed Team - Microsoft Corporation.
# Licensed under the MIT License.
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py


import time
from functools import partial
from typing import List, Optional

import torch

from archai.discrete_search.evaluators.torch_profiler_utils.hooks import (
    FLOPS,
    MACS,
    disable_functional_hooks,
    disable_tensor_hooks,
    enable_functional_hooks,
    enable_tensor_hooks,
)


class ProfilerModel:
    """Prepares a model to be used with profilling."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Initializes with custom arguments and keyword arguments.

        Args:
            model: Pre-trained model.

        """

        self.model = model
        self.is_profiling = False
        self.is_patched = False

    def start(self, ignore_layers: Optional[List[str]] = None) -> None:
        """Starts profiling.

        Args:
            ignore_layers: Layers to be ignored when profiling.

        """

        self.reset()

        enable_functional_hooks()
        enable_tensor_hooks()

        def register_hooks(module: torch.nn.Module, ignore_layers: List[str]) -> None:
            if ignore_layers and type(module) in ignore_layers:
                return

            def pre_hook(module: torch.nn.Module, input: torch.Tensor):
                FLOPS.append([])
                MACS.append([])

            if not hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                if FLOPS:
                    module.__flops__ += sum([elem[1] for elem in FLOPS[-1]])
                    FLOPS.pop()
                    module.__macs__ += sum([elem[1] for elem in MACS[-1]])
                    MACS.pop()

            if not hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__ = module.register_forward_hook(post_hook)

            def start_time_hook(module: torch.nn.Module, input: torch.Tensor):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                module.__start_time__ = time.time()

            if not hasattr(module, "__start_time_hook_handle"):
                module.__start_time_hook_handle__ = module.register_forward_pre_hook(start_time_hook)

            def end_time_hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                module.__latency__ += time.time() - module.__start_time__

            if not hasattr(module, "__end_time_hook_handle__"):
                module.__end_time_hook_handle__ = module.register_forward_hook(end_time_hook)

            def peak_memory_hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                if torch.cuda.is_available():
                    module.__peak_memory__ = torch.cuda.max_memory_allocated()
                    torch.cuda.reset_peak_memory_stats()

            if not hasattr(module, "__peak_memory_hook_handle__"):
                module.__peak_memory_hook_handle__ = module.register_forward_hook(peak_memory_hook)

        self.model.apply(partial(register_hooks, ignore_layers=ignore_layers))

        self.is_profiling = True
        self.is_patched = True

    def stop(self) -> None:
        """Stops profiling."""

        if self.is_profiling and self.is_patched:
            disable_functional_hooks()
            disable_tensor_hooks()

            self.is_patched = False

        def remove_hooks(module: torch.nn.Module) -> None:
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__
            if hasattr(module, "__start_time_hook_handle__"):
                module.__start_time_hook_handle__.remove()
                del module.__start_time_hook_handle__
            if hasattr(module, "__end_time_hook_handle__"):
                module.__end_time_hook_handle__.remove()
                del module.__end_time_hook_handle__
            if hasattr(module, "__peak_memory_hook_handle__"):
                module.__peak_memory_hook_handle__.remove()
                del module.__peak_memory_hook_handle__

        self.model.apply(remove_hooks)

    def reset(self) -> None:
        """Resets the profiler."""

        def reset_attrs(module: torch.nn.Module) -> None:
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters())
            module.__start_time__ = 0
            module.__latency__ = 0
            module.__peak_memory__ = 0

        self.model.apply(reset_attrs)

    def end(self) -> None:
        """Ends the profiler."""

        if not self.is_profiling:
            return

        self.stop()
        self.is_profiling = False

        def remove_attrs(module: torch.nn.Module) -> None:
            if hasattr(module, "__flops__"):
                del module.__flops__
            if hasattr(module, "__macs__"):
                del module.__macs__
            if hasattr(module, "__params__"):
                del module.__params__
            if hasattr(module, "__start_time__"):
                del module.__start_time__
            if hasattr(module, "__latency__"):
                del module.__latency__
            if hasattr(module, "__peak_memory__"):
                del module.__peak_memory__

        self.model.apply(remove_attrs)

    def get_flops(self) -> int:
        """Gets the model's number of FLOPs.

        Returns:
            (int): Number of floating point operations.

        """

        def _get(module: torch.nn.Module) -> int:
            flops = module.__flops__
            for child in module.children():
                flops += _get(child)
            return flops

        return _get(self.model)

    def get_macs(self) -> int:
        """Gets the model's number of MACs.

        Returns:
            (int): Number of multiply-accumulate operations.

        """

        def _get(module: torch.nn.Module) -> int:
            macs = module.__macs__
            for child in module.children():
                macs += _get(child)
            return macs

        return _get(self.model)

    def get_params(self) -> int:
        """Gets the model's total number of parameters.

        Returns:
            (int): Number of parameters.

        """

        return self.model.__params__

    def get_latency(self) -> float:
        """Gets the model's latency.

        Returns:
            (float): Latency (seconds).

        """

        def _get(module: torch.nn.Module) -> int:
            latency = module.__latency__
            if latency == 0:
                for child in module.children():
                    latency += child.__latency__
            return latency

        return _get(self.model)

    def get_peak_memory(self) -> float:
        """Gets the model's peak memory.

        Returns:
            (float): Peak memory (bytes).

        """

        def _get(module: torch.nn.Module) -> int:
            peak_memory = [module.__peak_memory__]
            for child in module.children():
                peak_memory += [_get(child)]
            return max(peak_memory)

        return _get(self.model)
