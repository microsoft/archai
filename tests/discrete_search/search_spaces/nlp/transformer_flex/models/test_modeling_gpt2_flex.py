# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from archai.discrete_search.search_spaces.nlp.transformer_flex.models.configuration_gpt2_flex import (
    GPT2FlexConfig,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.modeling_gpt2_flex import (
    GPT2FlexLMHeadModel,
    GPT2FlexModel,
)


@pytest.fixture
def config():
    return GPT2FlexConfig(vocab_size=128, n_embd=768, n_layer=2)


def test_gpt2_flex_lm_head_model_init(config):
    model = GPT2FlexLMHeadModel(config)

    # Assert that the model's transformer attribute has the correct type
    assert isinstance(model.transformer, GPT2FlexModel)
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert model.lm_head.in_features == config.n_embd
    assert model.lm_head.out_features == config.vocab_size


def test_gpt2_flex_lm_head_model_forward_pass(config):
    model = GPT2FlexLMHeadModel(config)

    # Assert that the model is able to forward pass
    input_tensor = torch.randint(0, config.vocab_size, (1, 32))
    output = model(input_tensor)
    assert output.logits.shape == (1, 32, config.vocab_size)
