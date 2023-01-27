# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from torch import nn
from torch.nn import functional as F

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.evaluators.nlp.parameters import (
    NonEmbeddingParamsProxy,
    TotalParamsProxy,
)


@pytest.fixture
def model():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embd = nn.Embedding(10, 10)
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = self.embd(x)
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return ArchaiModel(Model(), archid="test")


def test_total_params_proxy(model):
    # Assert that the number of trainable parameters is correct
    proxy = TotalParamsProxy(trainable_only=True)
    num_params = proxy.evaluate(model, None)
    assert num_params == sum(param.numel() for param in model.arch.parameters() if param.requires_grad)

    # Assert that the number of all parameters is correct
    proxy = TotalParamsProxy(trainable_only=False)
    num_params = proxy.evaluate(model, None)
    assert num_params == sum(param.numel() for param in model.arch.parameters())


def test_non_embedding_params_proxy(model):
    # Assert that the number of non-embedding trainable parameters is correct
    proxy = NonEmbeddingParamsProxy(trainable_only=True)
    non_embedding_params = proxy.evaluate(model, None)
    embedding_params = sum(param.numel() for param in model.arch.embd.parameters() if param.requires_grad)
    assert non_embedding_params + embedding_params == sum(
        param.numel() for param in model.arch.parameters() if param.requires_grad
    )

    # Assert that the number of non-embedding parameters is correct
    proxy = NonEmbeddingParamsProxy(trainable_only=False)
    non_embedding_params = proxy.evaluate(model, None)
    embedding_params = sum(param.numel() for param in model.arch.embd.parameters())
    assert non_embedding_params + embedding_params == sum(param.numel() for param in model.arch.parameters())
