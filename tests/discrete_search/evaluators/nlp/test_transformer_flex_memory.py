# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from archai.discrete_search.evaluators.nlp.transformer_flex_memory import (
    TransformerFlexOnnxMemory,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


@pytest.fixture
def search_space():
    return TransformerFlexSearchSpace("gpt2")


def test_transformer_flex_onnx_memory(search_space):
    arch = search_space.random_sample()
    objective = TransformerFlexOnnxMemory(search_space)

    # Assert that the returned memory is valid
    memory = objective.evaluate(arch, None)
    assert memory > 0.0
