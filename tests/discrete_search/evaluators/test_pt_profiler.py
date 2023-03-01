# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from archai.discrete_search.evaluators.pt_profiler import (
    TorchFlops,
    TorchLatency,
    TorchMacs,
    TorchNumParameters,
    TorchPeakCpuMemory,
    TorchPeakCudaMemory,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


@pytest.fixture
def search_space():
    return TransformerFlexSearchSpace("gpt2")


@pytest.fixture
def models(search_space):
    return [search_space.random_sample() for _ in range(5)]


@pytest.fixture
def sample_input():
    return torch.zeros(1, 1, 192, dtype=torch.long)


def test_torch_num_params(models):
    torch_num_params = TorchNumParameters()
    num_params = [torch_num_params.evaluate(model, None, None) for model in models]
    assert all(p > 0 for p in num_params)

    torch_num_params2 = TorchNumParameters(exclude_cls=[torch.nn.BatchNorm2d])
    num_params2 = [torch_num_params2.evaluate(model, None, None) for model in models]
    assert all(p > 0 for p in num_params2)

    torch_num_params3 = TorchNumParameters(exclude_cls=[torch.nn.BatchNorm2d], trainable_only=False)
    num_params3 = [torch_num_params3.evaluate(model, None, None) for model in models]
    assert all(p > 0 for p in num_params3)


def test_torch_flops(models, sample_input):
    torch_flops = TorchFlops(forward_args=sample_input)
    flops = [torch_flops.evaluate(model, None, None) for model in models]
    assert all(f > 0 for f in flops)


def test_torch_macs(models, sample_input):
    torch_macs = TorchMacs(forward_args=sample_input)
    macs = [torch_macs.evaluate(model, None, None) for model in models]
    assert all(m > 0 for m in macs)


def test_torch_latency(models, sample_input):
    torch_latency = TorchLatency(forward_args=sample_input, num_warmups=2, num_samples=2)
    latency = [torch_latency.evaluate(model, None, None) for model in models]
    assert all(lt > 0 for lt in latency)

    torch_latency2 = TorchLatency(forward_args=sample_input, num_warmups=0, num_samples=3, use_median=True)
    latency2 = [torch_latency2.evaluate(model, None, None) for model in models]
    assert all(lt > 0 for lt in latency2)


def test_torch_peak_cuda_memory(models, sample_input):
    if torch.cuda.is_available():
        torch_peak_memory = TorchPeakCudaMemory(forward_args=sample_input)
        peak_memory = [torch_peak_memory.evaluate(model, None, None) for model in models]
        assert all(m > 0 for m in peak_memory)


def test_torch_peak_cpu_memory(models, sample_input):
    torch_peak_memory = TorchPeakCpuMemory(forward_args=sample_input)
    peak_memory = [torch_peak_memory.evaluate(model, None, None) for model in models]
    assert all(m > 0 for m in peak_memory)
