# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PyTorch-based constraints.
"""

import math
from typing import List, Optional

import torch
import torch.utils.benchmark as benchmark
from torch.profiler import ProfilerActivity, profile

from archai.nlp.compression.quantization.ptq import dynamic_quantization_torch_from_model
from archai.nlp import train


def measure_torch_inference_latency(model: torch.nn.Module,
                                    use_quantization: Optional[bool] = False,
                                    use_median: Optional[bool] = False,
                                    batch_size: Optional[int] = 1,
                                    seq_len: Optional[int] = 192,
                                    n_threads: Optional[int] = 1,
                                    n_trials: Optional[int] = 10,
                                    device: Optional[str] = 'cpu') -> float:
    """Measures a model's inference latency.

    Args:
        model: Model instance.
        use_quantization: Whether latency should be calculated with quantizated model or not.
        use_median: Whether should use median instead of mean for latency measurement.
        batch_size: Batch size to measure the latency.
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
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.int64).to(device)
    
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
                              batch_size: Optional[int] = 1,
                              seq_len: Optional[int] = 192,
                              n_threads: Optional[int] = 1,
                              device: Optional[str] = 'cpu') -> float:
    """Measures a model's peak memory during inference.

    Args:
        model: Model instance.
        use_quantization: Whether peak memory should be calculated with quantizated model or not.
        batch_size: Batch size to measure the peak memory.
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
        'input_ids': torch.zeros((batch_size, seq_len), dtype=torch.int64).to(device)
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


def measure_torch_perplexity(model: torch.nn.Module) -> float:
    """Measures a model's validation perplexity.

    Args:
        model: Model instance.

    Returns:
        (float): Validation perplexity.

    """

    try:
        args, device = train.init()
    except:
        args, device = train.init(dlogger_enabled=False)

    args.data = '/home/gderosa/dataroot/textpred/wikitext-103'
    args.dataset = 'wt103'
    args.vocab = 'word'
    args.vocab_size = 10000
    args.refresh_cache = False
    args.ext_len = 0
    args.mem_len = 0
    args.tgt_len = 192
    args.eval_tgt_len = 192
    args.batch_size = 8
    args.eval_batch_size = 8
    args.batch_chunk = 1
    args.max_step = 25
    args.warmup_step = 5
    
    vocab, train_itr, valid_itr, _, file_stats = train.load_data(args, device)
    optimizer, optimizer_sparse = train.create_optimizer(args, model)
    scaler = train.create_grad_scaler(args, model, optimizer)
    para_model, model = train.distributed_model(args, model, device)
    model.to(device)
    scheduler, scheduler_sparse = train.create_scheduler(args, optimizer, optimizer_sparse)
    _, best_val_loss, _ = train.train_main(args, device, train_itr, valid_itr, model, para_model,
                                           None, optimizer, optimizer_sparse, scheduler,
                                           scheduler_sparse, scaler, vocab, file_stats[1])

    return math.exp(best_val_loss)
