from typing import Callable, Tuple
import psutil
import os
import tracemalloc
from torch.profiler import ProfilerActivity, profile
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

def inference_stats(model:nn.Module, **inputs)->Tuple(int, int):
    # return memory usage in bytes, cpu time in ms
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
        _ = model(**inputs)
    t = prof.key_averages()
    self_time, self_mem = 0, 0
    for ti in t:
        self_mem += ti.self_cpu_memory_usage
        self_time += ti.self_cpu_time_total
    return self_mem, self_time
