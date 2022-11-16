# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
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
    """"""

    assert isinstance(model, torch.nn.Module), "`model` must be a PyTorch model."

    profiler = ProfilerModel(model)
    model.eval()

    if model_args is None:
        model_args = ()

    for _ in range(n_warmups):
        with torch.no_grad():
            _ = model(*model_args, **model_kwargs)

    profiler.start_profile(ignore_layers=ignore_layers)
    with torch.no_grad():
        _ = model(*model_args, **model_kwargs)

    flops = profiler.get_total_flops()
    macs = profiler.get_total_macs()
    params = profiler.get_total_params()
    latency = profiler.get_total_duration()
    memory = profiler.get_total_memory()

    profiler.end_profile()

    return {
        "flops": flops,
        "macs": macs,
        "n_parameters": params,
        "latency": latency,
        "memory": memory
    }
