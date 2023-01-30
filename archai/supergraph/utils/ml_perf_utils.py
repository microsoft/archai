from typing import Callable, Tuple
import psutil
import os
import tracemalloc
import torch
from torch import profiler
from torch import nn
import gc

def model_memory(create_model:Callable[[], nn.Module])->Tuple[nn.Module, int]:
    # returns model and memory occupied by the model in process
    gc.collect()

    # baseline process memory
    process = psutil.Process(os.getpid())
    baseline_mem = process.memory_info().rss

    model = create_model()

    gc.collect()

    new_mem = process.memory_info().rss

    return model, new_mem-baseline_mem

def inference_stats(model:nn.Module, **inputs)->Tuple[int, int, int]:
    # return memory usage in bytes, cpu time in us
    # We basically sum "self" time of individual ops,
    # i.e., not including child time.
    # Pytorch also has record_function which gives
    # higher CPU time, probably because it includes
    # time spent other than ops.
    # Sometime profiler also generates [memory] node
    # which has negative value of memory.
    with torch.no_grad():
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_flops=True) as prof:
            with profiler.record_function('model_inference'):
                _ = model(**inputs)
    t = prof.key_averages()
    self_time, self_mem, flops, ti_memory, inf_cpu, inf_mem, inf_flops = 0, 0, 0, 0, 0, 0, 0
    for ti in t:
        if ti.key == '[memory]':
            ti_memory = -ti.self_cpu_memory_usage
            continue
        if ti.key == 'model_inference':
            inf_mem = -ti.cpu_memory_usage
            inf_cpu = ti.cpu_time_total
            inf_flops = ti.flops
            continue
        self_mem += ti.self_cpu_memory_usage
        self_time += ti.self_cpu_time_total
        flops += ti.flops
    return self_mem, self_time, flops, inf_cpu
