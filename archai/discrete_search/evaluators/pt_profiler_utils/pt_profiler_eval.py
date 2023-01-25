# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from archai.discrete_search.evaluators.pt_profiler_utils.pt_profiler_model import (
    ProfilerModel,
)


def profile(
    model: torch.nn.Module,
    model_args: Optional[Tuple[Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    num_warmups: Optional[int] = 1,
    num_samples: Optional[int] = 1,
    use_median: Optional[bool] = False,
    ignore_layers: Optional[List[str]] = None,
) -> Dict[str, Union[float, int]]:
    """Profiles a PyTorch model.

    Outputs FLOPs, MACs, number of parameters, latency and peak memory.

    Args:
        model: PyTorch model.
        model_args: `model.forward` arguments.
        model_kwargs: `model.forward` keyword arguments.
        num_warmups: Number of warmup runs before profilling.
        num_samples: Number of runs after warmup.
        use_median: Whether to use median instead of mean to average memory and latency.
        ignore_layers: List of layer names that should be ignored during profiling.

    Returns:
        FLOPs, MACs, number of parameters, latency (seconds) and peak memory (bytes).

    """

    assert isinstance(model, torch.nn.Module), "`model` must be a PyTorch model."
    model_kwargs = model_kwargs or dict()

    profiler = ProfilerModel(model)
    model.eval()

    if model_args is None:
        model_args = ()

    for _ in range(num_warmups):
        with torch.no_grad():
            _ = model(*model_args, **model_kwargs)

    result = {"flops": None, "macs": None, "n_parameters": None, "latency": [], "peak_memory": []}

    for _ in range(num_samples):
        profiler.start(ignore_layers=ignore_layers)

        with torch.no_grad():
            _ = model(*model_args, **model_kwargs)

        result.update(
            {"flops": profiler.get_flops(), "macs": profiler.get_macs(), "n_parameters": profiler.get_params()}
        )

        result["latency"].append(profiler.get_latency())
        result["peak_memory"].append(profiler.get_peak_memory())

        profiler.end()

    stat = statistics.median if use_median else statistics.mean

    result["latency"] = stat(result["latency"])
    result["peak_memory"] = stat(result["peak_memory"])

    return result
