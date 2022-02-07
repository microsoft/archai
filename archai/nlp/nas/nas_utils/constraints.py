# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constraints that are used throughout the search procedure.
"""

from typing import List, Optional

import torch
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile

from archai.nlp.compression.quantization.ptq import dynamic_quantization_torch_from_model

# Latency upper bound on different device targets
# Any model with more latency than this will be removed from consideration during search
DEVICE_LATENCY_CONSTRAINT = {
    'XeonE5-2690': 10.0,
    'corei7': 10.0,
    'corei5': 10.0,
    'D3_V2': 10.0,
}


def measure_inference_latency(model: torch.nn.Module,
                              use_quantization: Optional[bool] = False,
                              use_median: Optional[bool] = False,
                              seq_len: Optional[int] = 192,
                              n_threads: Optional[int] = 1,
                              n_trials: Optional[int] = 10,
                              device: Optional[str] = 'cpu') -> float:
    """Measures a model's inference latency.

    Args:
        model: Model instance.
        use_quantization: Whether latency should be calculated with quantizated model or not.
        use_median: Whether should use median instead of mean for latency measurement.
        seq_len: Sequence length to measure the latency.
        n_threads: Number of inference threads.
        n_trials: Number of times to repeat the measurement.
        device: Device where latency should be measured.

    Returns:
        (float): Mean or median latency in seconds.

    """

    if use_quantization:
        dynamic_quantization_torch_from_model(model)

    torch.set_num_threads(n_threads)

    model = model.to(device=device)
    input_ids = torch.zeros((1, seq_len), dtype=torch.int64).to(device)
    
    timer = benchmark.Timer(stmt='model(input_ids, labels, mems)',
                            globals={
                                'input_ids': input_ids,
                                'labels': None,
                                'mems': None,
                                'model': model
                            },
                            num_threads=n_threads)

    runner = timer.timeit(n_trials)

    return runner.median if use_median else runner.mean


def measure_parameters(model: torch.nn.Module,
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


def measure_peak_memory(model: torch.nn.Module,
                        use_quantization: Optional[bool] = False,
                        seq_len: Optional[int] = 192,
                        n_threads: Optional[int] = 1,
                        device: Optional[str] = 'cpu') -> float:
    """Measures a model's peak memory during inference.

    Args:
        model: Model instance.
        use_quantization: Whether latency should be calculated with quantizated model or not.
        seq_len: Sequence length to measure the peak memory.
        n_threads: Number of inference threads.
        device: Device where peak memory should be measured.

    Returns:
        (float): Peak memory during inference in megabytes.

    """

    if use_quantization:
        dynamic_quantization_torch_from_model(model)

    torch.set_num_threads(n_threads)

    model = model.to(device=device)
    inputs = {
        'input_ids': torch.zeros((1, seq_len), dtype=torch.int64).to(device)
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
    peak_memory_mb = peak_memory / (1024*1024)

    return peak_memory_mb
