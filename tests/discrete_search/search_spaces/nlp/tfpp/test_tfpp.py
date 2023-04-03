# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import torch
import pytest
from transformers import PretrainedConfig, CodeGenConfig

from archai.discrete_search.api import ArchaiModel
from archai.discrete_search.search_spaces.config import ArchConfig, ConfigSearchSpace
from archai.discrete_search.search_spaces.nlp import TfppSearchSpace

N_POSITIONS = 2048

def check_fwd_pass(model: ArchaiModel):
    assert isinstance(model, ArchaiModel)
    
    arch_config = model.metadata['config']
    assert isinstance(arch_config, ArchConfig)
    
    _ = arch_config.pick('hidden_size', record_usage=False)
    x = torch.randint(high=10, size=(1, N_POSITIONS))

    # Test forward pass
    y = model.arch(x)


def test_tfpp_sample_codegen():
    search_space = TfppSearchSpace(
        'codegen', total_layers=[1], total_heads=[6],
        homogeneous=True, seed=1, n_positions=N_POSITIONS
    )

    for _ in range(5):
        model = search_space.random_sample()
        check_fwd_pass(model)


def test_tfpp_sample_gpt2():
    search_space = TfppSearchSpace(
        'gpt2', total_layers=[1], total_heads=[6],
        homogeneous=True, seed=1, n_positions=N_POSITIONS
    )

    for _ in range(5):
        model = search_space.random_sample()
        check_fwd_pass(model)
