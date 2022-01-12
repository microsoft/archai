# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Defines constraints that are used throughout the search space.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.utils.benchmark as benchmark

from archai.nlp.models.mem_transformer.model_mem_transformer import MemTransformerLM


def get_model(model_config, train=False):
    model = MemTransformerLM(**model_config)

    if not train:
        model = model.to(device='cpu')
        model.eval()

    return model


def get_latency(model, model_config, n_threads=1, repeat=10):
    if n_threads > 1:
        torch.set_num_threads(n_threads)

    model = model.to(device='cpu')

    t0 = benchmark.Timer(stmt='model(input_ids, labels, mems)',
                         setup='',
                         globals={'input_ids': torch.LongTensor(model_config['tgt_len']).random_(0, model_config['n_token']).unsqueeze(0), 'labels': None, 'mems': None, 'model': model},
                         num_threads=n_threads,
                         label='Multithreaded model execution')

    info = t0.timeit(repeat)
    info._lazy_init()

    latency = info._mean

    return latency
    