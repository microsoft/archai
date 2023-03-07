# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from archai.discrete_search.evaluators.nlp.transformer_flex_latency import (
    TransformerFlexOnnxLatency,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


@pytest.fixture
def search_space():
    return TransformerFlexSearchSpace("gpt2")


def test_transformer_flex_onnx_latency(search_space):
    arch = search_space.random_sample()
    objective = TransformerFlexOnnxLatency(search_space)

    # Assert that the returned latency is valid
    latency = objective.evaluate(arch)
    assert latency > 0.0
