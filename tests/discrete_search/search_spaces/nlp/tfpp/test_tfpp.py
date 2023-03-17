# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import torch
import pytest
from transformers import PretrainedConfig, CodeGenConfig

from archai.discrete_search.api import ArchaiModel
from archai.discrete_search.search_spaces.config import ArchConfig, ConfigSearchSpace
from archai.discrete_search.search_spaces.nlp import TfppSearchSpace


@pytest.fixture
def hf_config() -> PretrainedConfig:
    return CodeGenConfig()


def check_fwd_pass(hf_config: PretrainedConfig, model: ArchaiModel):
    assert isinstance(model, ArchaiModel)
    
    arch_config = model.metadata['config']
    assert isinstance(arch_config, ArchConfig)
    
    hidden_size = arch_config.pick('hidden_size', record_usage=False)
    x = torch.randint(high=10, size=(1, hf_config.n_positions))

    # Test forward pass
    y = model.arch(x)


def test_tfpp_sample(hf_config):
    search_space = TfppSearchSpace(
        hf_config, total_layers=[1], total_heads=[6],
        homogeneous=True, seed=1
    )

    for _ in range(5):
        model = search_space.random_sample()
        check_fwd_pass(hf_config, model)

