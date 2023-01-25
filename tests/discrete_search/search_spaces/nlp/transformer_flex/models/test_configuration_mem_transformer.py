# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.discrete_search.search_spaces.nlp.transformer_flex.models.configuration_mem_transformer import (
    MemTransformerConfig,
)


def test_mem_transformer_config():
    # Assert that the config has the correct values
    config = MemTransformerConfig()

    assert config.model_type == "mem-transformer"
    assert not config.primer_conv
    assert not config.primer_square
    assert not config.fp16
    assert not config.use_cache
