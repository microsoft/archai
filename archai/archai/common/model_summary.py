# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Mapping, Sized, Sequence
import math

import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np

from numbers import Number


def summary(model, input_size):
    result, params_info = summary_string(model, input_size)
    print(result)

    return params_info

def is_scaler(o):
    return isinstance(o, Number) or isinstance(o, str) or o is None


def get_tensor_stat(tensor):
    assert isinstance(tensor, torch.Tensor)

    # some pytorch low-level memory management constant
    # the minimal allocate memory size (Byte)
    PYTORCH_MIN_ALLOCATE = 2 ** 9
    # the minimal cache memory size (Byte)
    PYTORCH_MIN_CACHE = 2 ** 20

    numel = tensor.numel()
    element_size = tensor.element_size()
    fact_numel = tensor.storage().size()
    fact_memory_size = fact_numel * element_size
    # since pytorch allocate at least 512 Bytes for any tensor, round
    # up to a multiple of 512
    memory_size = math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) \
            * PYTORCH_MIN_ALLOCATE

    # tensor.storage should be the actual object related to memory
    # allocation
    data_ptr = tensor.storage().data_ptr()
    size = tuple(tensor.size())
    # torch scalar has empty size
    if not size:
        size = (1,)

    return ([size], numel, memory_size)

def get_all_tensor_stats(o):
    if is_scaler(o):
        return ([[]], 0, 0)
    elif isinstance(o, torch.Tensor):
        return get_tensor_stat(o)
    elif isinstance(o, Mapping):
        return get_all_tensor_stats(o.values())
    elif isinstance(o, Iterable): # tuple, list, maps
        stats = [[]], 0, 0
        for oi in o:
            tz = get_all_tensor_stats(oi)
            stats = tuple(x+y for x,y in zip(stats, tz))
        return stats
    elif hasattr(o, '__dict__'):
        return get_all_tensor_stats(o.__dict__)
    else:
        return ([[]], 0, 0)


def get_shape(o):
    if is_scaler(o):
        return str(o)
    elif hasattr(o, 'shape'):
        return f'shape{o.shape}'
    elif hasattr(o, 'size'):
        return f'size{o.size()}'
    elif isinstance(o, Sequence):
        if len(o)==0:
            return 'seq[]'
        elif is_scaler(o[0]):
            return f'seq[{len(o)}]'
        return f'seq{[get_shape(oi) for oi in o]}'
    elif isinstance(o, Mapping):
        if len(o)==0:
            return 'map[]'
        elif is_scaler(next(o)):
            return f'map[{len(o)}]'
        arr = [(get_shape(ki), get_shape(vi)) for ki, vi in o]
        return f'map{arr}'
    else:
        return 'N/A'



def summary_string(model, input_size, dtype=torch.float32):
    summary_str = ''

    # create properties
    summary = OrderedDict()
    hooks = []

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)

            summary[m_key] = OrderedDict()
            summary[m_key]["input"] = get_all_tensor_stats(input)
            summary[m_key]["output"] = get_all_tensor_stats(output)

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size()))).item()
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size()))).item()
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # batch_size of 2 for batchnorm
    x = torch.rand(input_size, dtype=dtype,
                   device=next(model.parameters()).device)

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output (elments, mem)", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_input = get_tensor_stat(x)
    total_output = ([[], 0, 0])
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output"][1:]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output = tuple(x+y for x,y in zip(total_output, summary[layer]["output"]))
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    total_numel = total_params + total_output[1] + total_input[1]

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += f"Input Elments: {total_input[1]:.4e}\n"
    summary_str += f"Input Mem: {total_input[2]:.4e}\n"
    summary_str += f"Layer Output Elements: {total_output[1]:.4e}\n"
    summary_str += f"Layer Output Mem: {total_output[2]:.4e}\n"
    summary_str += f"Params {total_params:.4e}\n"
    summary_str += f"Total Elements {total_numel:.4e}\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)