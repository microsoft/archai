# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from archai.quantization.mixed_qat import MixedQAT


@pytest.fixture
def base_model():
    return GPT2LMHeadModel(config=GPT2Config(n_layer=1, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))


def test_mixed_qat_init(base_model):
    # Assert that the `qat_weight` parameter is checked and a ValueError is raised
    # if it is not between 0 and 1
    with pytest.raises(ValueError):
        MixedQAT(base_model, qat_weight=-0.1)
    with pytest.raises(ValueError):
        MixedQAT(base_model, qat_weight=1.1)

    # Assert that the `model` and `qat_model` attributes are correctly initialized
    mixed_qat = MixedQAT(base_model)
    assert mixed_qat.model is base_model
    assert mixed_qat.qat_model is not base_model

    # Assert that the parameters of the `model` and `qat_model` are correctly shared
    for param, qat_param in zip(base_model.parameters(), mixed_qat.qat_model.parameters()):
        assert qat_param is param


def test_mixed_qat_forward(base_model):
    mixed_qat = MixedQAT(base_model)
    x = torch.zeros((1, 192), dtype=torch.long)

    # Assert that the `qat_outputs` are returned when model is not in training mode
    mixed_qat.eval()
    qat_outputs = mixed_qat.qat_model(input_ids=x, labels=x)
    assert qat_outputs.loss == mixed_qat(input_ids=x, labels=x)[0]

    # Assert that the linear combination of losses is returned when model is in training mode
    mixed_qat.train()
    outputs = mixed_qat.model(input_ids=x, labels=x)
    qat_outputs = mixed_qat.qat_model(input_ids=x, labels=x)
    assert (
        outputs.loss * mixed_qat.regular_weight + qat_outputs.loss * mixed_qat.qat_weight
        == mixed_qat(input_ids=x, labels=x)[0]
    )
