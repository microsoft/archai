# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Profiler-based evaluation.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from archai.nlp.eval.profiler.profiler_model import ProfilerModel


def evaluate(
    model: torch.nn.Module,
    model_args: Optional[Tuple[Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    n_warmups: Optional[int] = 1,
    ignore_layers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Performs the profiler-based evaluation.

    Args:
        model: PyTorch-based model.
        model_args: Model's forward arguments.
        model_kwargs: Model's forward keyword arguments.
        n_warmups: Number of warmups before profiling.
        ignore_layers: Layers that should be ignored during profiling.

    Returns:
        (Dict[str, Any]): FLOPs, MACs, number of parameters,
            latency (seconds) and peak memory (bytes).

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
