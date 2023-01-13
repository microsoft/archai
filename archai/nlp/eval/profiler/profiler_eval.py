# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Profiler-based evaluation."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from archai.nlp.eval.profiler.profiler_model import ProfilerModel


def profile(
    model: torch.nn.Module,
    model_args: Optional[Tuple[Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    n_warmups: Optional[int] = 1,
    ignore_layers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Profiles the performance of a PyTorch model.

    Args:
        model: The PyTorch model to evaluate.
        model_args: The forward arguments for the model.
        model_kwargs: The forward keyword arguments for the model.
        n_warmups: The number of warmup iterations to run before profiling.
        ignore_layers: A list of layer names to ignore during profiling.

    Returns:
        A dictionary containing the following performance metrics:
            - flops: The number of floating point operations (FLOPs) performed by the model.
            - macs: The number of multiply-accumulate operations (MACs) performed by the model.
            - n_parameters: The number of parameters in the model.
            - latency: The latency of the model, in seconds.
            - peak_memory: The peak memory usage of the model, in bytes.

    """

    assert isinstance(model, torch.nn.Module), "`model` must be a PyTorch model."

    profiler = ProfilerModel(model)
    model.eval()

    if model_args is None:
        model_args = ()

    for _ in range(n_warmups):
        with torch.no_grad():
            _ = model(*model_args, **model_kwargs)

    profiler.start(ignore_layers=ignore_layers)

    with torch.no_grad():
        _ = model(*model_args, **model_kwargs)

    flops = profiler.get_flops()
    macs = profiler.get_macs()
    params = profiler.get_params()
    latency = profiler.get_latency()
    peak_memory = profiler.get_peak_memory()

    profiler.end()

    return {"flops": flops, "macs": macs, "n_parameters": params, "latency": latency, "peak_memory": peak_memory}
