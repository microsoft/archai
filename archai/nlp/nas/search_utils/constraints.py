# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Defines constraints that are used throughout the search procedure.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.benchmark as benchmark
from memory_profiler import memory_usage


def measure_inference_latency(model: torch.nn.Module,
                              use_median: Optional[bool] = False,
                              seq_len: Optional[int] = 192,
                              n_threads: Optional[int] = 1,
                              n_trials: Optional[int] = 10,
                              device: Optional[str] = 'cpu') -> float:
    """Measures a model's inference latency.

    Args:
        model: Model instance.
        use_median: Whether should use median instead of mean for latency measurement.
        seq_len: Sequence length to measure the latency.
        n_threads: Number of inference threads.
        n_trials: Number of times to repeat the measurement.
        device: Device where latency should be measured.

    Returns:
        (float): Mean or median latency.

    """

    torch.set_num_threads(n_threads)

    model = model.to(device=device)
    timer = benchmark.Timer(stmt='model(input_ids, labels, mems)',
                            globals={
                                'input_ids': torch.zeros((1, seq_len), dtype=torch.int64),
                                'labels': None,
                                'mems': None,
                                'model': model
                            },
                            num_threads=n_threads)

    runner = timer.timeit(n_trials)

    return runner.median if use_median else runner.mean


def measure_parameters(model: torch.nn.Module,
                       layers: Optional[List[str]] = ['attention', 'ff']) -> int:
    """Measures a model's number of parameters according to input options.

    Args:
        model: Model instance.
        layers: Layers that should be used in measurement.

    Returns:
        (int): Number of parameters.

    """

    params = model.get_params()

    return sum([params[l] for l in layers])


def measure_peak_memory(model: torch.nn.Module,
                        use_median: Optional[bool] = False,
                        seq_len: Optional[int] = 192,
                        n_threads: Optional[int] = 1,
                        n_trials: Optional[int] = 10,
                        device: Optional[str] = 'cpu') -> float:
    """Measures a model's peak memory.

    Args:
        model: Model instance.
        use_median: Whether should use median instead of mean for peak memory measurement.
        seq_len: Sequence length to measure the peak memory.
        n_threads: Number of inference threads.
        n_trials: Number of times to repeat the measurement.
        device: Device where peak memory should be measured.

    Returns:
        (float): Mean or median peak memory.

    """

    def _track_peak_memory(model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> None:
        return model(**inputs)

    torch.set_num_threads(n_threads)

    model = model.to(device=device)
    inputs = {
        'input_ids': torch.zeros((1, seq_len), dtype=torch.int64).to(device)
    }

    peak_memory = []
    for _ in range(n_trials):
        rss = memory_usage(proc=(_track_peak_memory, (model, inputs)),
                           max_usage=True,
                           backend='psutil',
                           include_children=False,
                           multiprocess=True)
        peak_memory.append(rss)

    return np.median(peak_memory) if use_median else np.mean(peak_memory)
