# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from archai.nlp.objectives.transformer_flex_memory import TransformerFlexOnnxMemory
from archai.nlp.search_spaces.transformer_flex.search_space import (
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
