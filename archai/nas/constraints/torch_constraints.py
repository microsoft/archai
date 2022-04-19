# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PyTorch-based constraints.
"""

from argparse import Namespace
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile

#from archai.nlp import train
#from archai.nlp.compression.quantization.ptq import dynamic_quantization_torch_from_model
#from archai.nlp.metrics.text_predict.predictor import run_score


def measure_torch_inference_latency(model: torch.nn.Module,
                                    use_quantization: Optional[bool] = False,
                                    use_median: Optional[bool] = False,
                                    input_dims: Optional[Tuple[int]] = (1, 3, 32, 32),
                                    n_threads: Optional[int] = 1,
                                    n_trials: Optional[int] = 10,
                                    device: Optional[str] = 'cpu') -> float:
    """Measures a model's inference latency.

    Args:
        model: Model instance.
        use_quantization: (TODO) Whether latency should be calculated with quantizated model or not.
        use_median: Whether should use median instead of mean for latency measurement.
        input_dims: Size of input data which will be put through the model.
        n_threads: Number of inference threads.
        n_trials: Number of times to repeat the measurement.
        device: Device where latency should be measured.

    Returns:
        (float): Mean or median latency in seconds.

    """

    # if use_quantization:
    #     dynamic_quantization_torch_from_model(model)        

    torch.set_num_threads(n_threads)

    model = model.to(device=device)
    input = torch.zeros(input_dims, dtype=torch.float32).to(device)
    
    timer = benchmark.Timer(stmt='model(input)',
                            globals={
                                'input': input,                                
                                'model': model
                            },
                            num_threads=n_threads)

    runner = timer.timeit(n_trials)

    return runner.median if use_median else runner.mean


def measure_torch_parameters(model: torch.nn.Module,
                             keys: Optional[List[str]] = ['non_embedding']) -> int:
    """Measures a model's number of parameters according to input options.

    Args:
        model: Model instance.
        keys: Keys that should be used in measurement.

    Returns:
        (int): Number of parameters.

    """

    params = model.get_params()

    return sum([params[l] for l in keys])


def measure_torch_peak_memory(model: torch.nn.Module,
                              use_quantization: Optional[bool] = False,
                              input_dims: Optional[Tuple[int]] = (1, 3, 32, 32),
                              n_threads: Optional[int] = 1,
                              device: Optional[str] = 'cpu') -> float:
    """Measures a model's peak memory during inference.

    Args:
        model: Model instance.
        use_quantization: Whether peak memory should be calculated with quantizated model or not.
        input_dims: Size of input data which will be put through the model.
        n_threads: Number of inference threads.
        device: Device where peak memory should be measured.

    Returns:
        (float): Peak memory during inference in megabytes.

    """

    # if use_quantization:
    #     dynamic_quantization_torch_from_model(model)

    torch.set_num_threads(n_threads)

    model = model.to(device=device)
    inputs = {
        'input_ids': torch.zeros(input_dims, dtype=torch.float32).to(device)
    }

    if device == 'cpu':
        activities = [ProfilerActivity.CPU]
        device_key = 'self_cpu_memory_usage'
    else:
        activities = [ProfilerActivity.CUDA]
        device_key = 'cuda_memory_usage'

    with profile(activities=activities, profile_memory=True) as p:
        model(**inputs)

    peak_memory = sum([getattr(key, device_key) if getattr(key, device_key) > 0 else 0
                       for key in p.key_averages()])
    peak_memory_mb = peak_memory / (1024 ** 2)

    return peak_memory_mb