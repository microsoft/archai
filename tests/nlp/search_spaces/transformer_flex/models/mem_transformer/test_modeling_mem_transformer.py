# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.configuration_mem_transformer import (
    MemTransformerConfig,
)
from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.modeling_mem_transformer import (
    MemTransformerLMHeadModel,
    MemTransformerModel,
)
from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.utils.projected_adaptive_log_softmax import (
    ProjectedAdaptiveLogSoftmax,
)


@pytest.fixture
def config():
    return MemTransformerConfig(vocab_size=128, d_embed=1024, d_model=1024, cutoffs=[32, 64], div_val=1)


def test_mem_transformer_lm_head_model_init(config):
    model = MemTransformerLMHeadModel(config)

    # Assert that the model's transformer attribute has the correct type
    assert isinstance(model.transformer, MemTransformerModel)
    assert isinstance(model.crit, ProjectedAdaptiveLogSoftmax)
    assert model.crit.d_embed == config.d_embed
    assert model.crit.d_model == config.d_model
    assert model.crit.vocab_size == config.vocab_size


def test_mem_transformer_lm_head_model_forward_pass(config):
    model = MemTransformerLMHeadModel(config)

    # Assert that the model is able to forward pass
    input_tensor = torch.randint(0, config.vocab_size, (1, 32))
    output = model(input_tensor)
    assert output.prediction_scores.shape == (1, 32, config.vocab_size)
