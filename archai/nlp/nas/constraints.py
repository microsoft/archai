# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Defines constraints that are used throughout the search space.
"""

from typing import Optional

import torch
import torch.utils.benchmark as benchmark


def measure_latency(model: torch.nn.Module,
                    n_threads: Optional[int] = 1,
                    n_trials: Optional[int] = 10) -> float:
    """Measures a model's inference latency.

    Args:
        model: Model instance.
        n_threads: Number of inference threads.
        n_trials: Number of times to repeat the measurement.

    Returns:
        (float): Mean latency.

    """

    if n_threads > 1:
        torch.set_num_threads(n_threads)

    model = model.to(device='cpu')

    timer = benchmark.Timer(stmt='model(input_ids, labels, mems)',
                            setup='',
                            globals={'input_ids': torch.zeros((1, 192), dtype=torch.int64), 'labels': None, 'mems': None, 'model': model},
                            num_threads=n_threads,
                            label='Multithreaded model execution')

    timer_runner = timer.timeit(n_trials)
    timer_runner._lazy_init()

    latency = timer_runner._mean

    return latency
    