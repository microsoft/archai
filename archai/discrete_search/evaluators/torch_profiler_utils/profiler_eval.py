# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Tuple, Union
import statistics

import torch

from archai.discrete_search.evaluators.torch_profiler_utils.profiler_model import ProfilerModel


def profile(
    model: torch.nn.Module,
    model_args: Optional[Tuple[Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    num_warmups: Optional[int] = 1,
    num_samples: Optional[int] = 1,
    use_median: Optional[bool] = False,
    ignore_layers: Optional[List[str]] = None,
) -> Dict[str, Union[float, int]]:
    """Profiles a PyTorch model. Outputs FLOPs, MACs, number of parameters, latency and peak memory.

    Args:
        model (torch.nn.Module): PyTorch model.
        model_args (Optional[Tuple[Any]], optional): `model.forward` arguments. Defaults to None.
        model_kwargs (Optional[Dict[str, Any]], optional): `model.forward` keyword arguments. Defaults to None.
        num_warmups (int, optional): Number of warmup runs before profilling. Defaults to 1.
        num_samples (int, optional): Number of runs after warmup. Defaults to 1
        use_median (bool, optional): Whether to use median instead of mean to average memory and latency. Defaults to False.
        ignore_layers (Optional[List[str]], optional): List of layer names that should be ignored during 
            profiling. Defaults to None

    Returns:
        Dict[str, Any]: FLOPs, MACs, number of parameters, latency (seconds) and peak memory (bytes).
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

    result = {
        'flops': None, 'macs': None, 'n_parameters': None,
        'latency': [], 'peak_memory': []
    }

    for _ in range(num_samples):
        profiler.start(ignore_layers=ignore_layers)

        with torch.no_grad():
            _ = model(*model_args, **model_kwargs)

        result.update({
            'flops': profiler.get_flops(),
            'macs': profiler.get_macs(),
            'n_parameters': profiler.get_params()
        })

        result['latency'].append(profiler.get_latency())
        result['peak_memory'].append(profiler.get_peak_memory())
        
        profiler.end()

    stat = statistics.median if use_median else statistics.mean
    
    result['latency'] = stat(result['latency'])
    result['peak_memory'] = stat(result['peak_memory'])
    
    return result
