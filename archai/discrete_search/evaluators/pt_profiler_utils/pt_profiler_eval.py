# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import statistics
from typing import Any, Dict, List, Optional, Union

import torch

from archai.discrete_search.evaluators.pt_profiler_utils.pt_profiler_model import (
    ProfilerModel,
)


def profile(
    model: torch.nn.Module,
    forward_args: Optional[List[Any]] = None,
    forward_kwargs: Optional[Dict[str, Any]] = None,
    num_warmups: Optional[int] = 1,
    num_samples: Optional[int] = 1,
    use_cuda: Optional[bool] = False,
    use_median: Optional[bool] = False,
    ignore_layers: Optional[List[str]] = None,
) -> Dict[str, Union[float, int]]:
    """Profile a PyTorch model.

    Outputs FLOPs, MACs, number of parameters, latency and peak memory.

    Args:
        model: PyTorch model.
        forward_args: `model.forward()` arguments used for profilling.
        forward_kwargs: `model.forward()` keyword arguments used for profilling.
        num_warmups: Number of warmup runs before profilling.
        num_samples: Number of runs after warmup.
        use_cuda: Whether to use CUDA instead of CPU.
        use_median: Whether to use median instead of mean to average memory and latency.
        ignore_layers: List of layer names that should be ignored during profiling.

    Returns:
        FLOPs, MACs, number of parameters, latency (seconds) and peak memory (bytes).

    """

    assert isinstance(model, torch.nn.Module), "`model` must be a PyTorch model."
    forward_args = forward_args if forward_args is not None else []
    forward_args = [forward_args] if isinstance(forward_args, torch.Tensor) else forward_args
    forward_kwargs = forward_kwargs or {}

    if use_cuda:
        # Ensures that model and all inputs are put on CUDA before profiling
        model.to("cuda")
        forward_args = tuple([arg.to("cuda") for arg in forward_args])
        forward_kwargs = {key: value.to("cuda") for key, value in forward_kwargs.items()}

    profiler = ProfilerModel(model)
    model.eval()

    for _ in range(num_warmups):
        with torch.no_grad():
            _ = model(*forward_args, **forward_kwargs)

    result = {"flops": None, "macs": None, "n_parameters": None, "latency": [], "peak_memory": []}

    for _ in range(num_samples):
        profiler.start(ignore_layers=ignore_layers)

        with torch.no_grad():
            _ = model(*forward_args, **forward_kwargs)

        result.update(
            {"flops": profiler.get_flops(), "macs": profiler.get_macs(), "n_parameters": profiler.get_params()}
        )

        result["latency"].append(profiler.get_latency())
        result["peak_memory"].append(profiler.get_peak_memory())

        profiler.end()

    if use_cuda:
        # Ensures that model and all inputs are put on CPU after profiling to avoid
        # overloading GPU memory
        model.to("cpu")
        forward_args = tuple([arg.to("cpu") for arg in forward_args])
        forward_kwargs = {key: value.to("cpu") for key, value in forward_kwargs.items()}

    stat = statistics.median if use_median else statistics.mean

    result["latency"] = stat(result["latency"])
    result["peak_memory"] = stat(result["peak_memory"])

    return result
