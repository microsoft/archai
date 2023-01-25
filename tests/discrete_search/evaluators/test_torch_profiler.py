# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from archai.discrete_search.evaluators.pt_profiler import (
    TorchCudaPeakMemory,
    TorchFlops,
    TorchLatency,
    TorchMacs,
    TorchNumParameters,
)
from archai.discrete_search.search_spaces.segmentation_dag.search_space import (
    SegmentationDagSearchSpace,
)


@pytest.fixture
def search_space():
    return SegmentationDagSearchSpace(nb_classes=1, img_size=(96, 96), seed=42)


@pytest.fixture
def models(search_space):
    return [search_space.random_sample() for _ in range(10)]


@pytest.fixture
def sample_input():
    return torch.randn(1, 3, 96, 96)


def test_torch_num_params(models, sample_input):
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
    torch_flops = TorchFlops(sample_args=(sample_input,))
    flops = [torch_flops.evaluate(model, None, None) for model in models]
    assert all(f > 0 for f in flops)


def test_torch_macs(models, sample_input):
    torch_macs = TorchMacs(sample_args=(sample_input,))
    macs = [torch_macs.evaluate(model, None, None) for model in models]
    assert all(m > 0 for m in macs)


def test_torch_latency(models, sample_input):
    torch_latency = TorchLatency(sample_args=(sample_input,), num_warmups=2, num_samples=2)
    latency = [torch_latency.evaluate(model, None, None) for model in models]
    assert all(lt > 0 for lt in latency)

    torch_latency2 = TorchLatency(sample_args=(sample_input,), num_warmups=0, num_samples=3, use_median=True)
    latency2 = [torch_latency2.evaluate(model, None, None) for model in models]
    assert all(lt > 0 for lt in latency2)


def test_torch_peak_memory(models, sample_input):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_peak_memory = TorchCudaPeakMemory(sample_args=(sample_input.to(device),))

    for model in models:
        model.arch.to(device)

    peak_memory = [torch_peak_memory.evaluate(model, None, None) for model in models]
    assert all((m > 0 if device == "cuda" else m == 0) for m in peak_memory)
