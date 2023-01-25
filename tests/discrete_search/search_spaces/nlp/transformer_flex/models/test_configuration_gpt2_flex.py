# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.discrete_search.search_spaces.nlp.transformer_flex.models.configuration_gpt2_flex import (
    GPT2FlexConfig,
)


def test_gpt2_flex_config():
    # Assert that the config has the correct values
    config = GPT2FlexConfig(n_layer=3, primer_square=True)

    assert config.model_type == "gpt2-flex"
    assert config.primer_square is True
    assert config.activation_function == "relu"

    assert config.n_inner is not None
    assert config.n_inner == [4 * config.n_embd for _ in range(config.n_layer)]

    assert config.n_head is not None
    assert config.n_head == [12 for _ in range(config.n_layer)]
