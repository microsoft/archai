# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional, Union

import torch
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.evaluators.pt_profiler_utils.pt_profiler_eval import profile


class TorchNumParameters(ModelEvaluator):
    """Total number of parameters."""

    def __init__(
        self, exclude_cls: Optional[List[torch.nn.Module]] = None, trainable_only: Optional[bool] = True
    ) -> None:
        """Initialize the evaluator.

        Args:
            exclude_cls: List of PyTorch module classes to exclude from parameter counting.
            trainable_only: A flag indicating whether only trainable parameters should be counted.

        """

        self.exclude_cls = exclude_cls
        self.trainable_only = trainable_only

    @overrides
    def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        total_params = sum(
            param.numel() for param in model.arch.parameters() if not self.trainable_only or param.requires_grad
        )

        exclude_params = (
            0
            if self.exclude_cls is None
            else sum(
                sum(param.numel() for param in module.parameters())
                for module in model.arch.modules()
                if isinstance(module, tuple(self.exclude_cls))
            )
        )

        return total_params - exclude_params


class TorchFlops(ModelEvaluator):
    """Total number of FLOPs."""

    def __init__(
        self,
        forward_args: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        forward_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ignore_layers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            forward_args: `model.forward()` arguments used for profilling.
            forward_kwargs: `model.forward()` keyword arguments used for profilling.
            ignore_layers: List of layer names that should be ignored during profiling.

        """

        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, budget: Optional[float] = None) -> float:
        return profile(
            model.arch,
            self.forward_args,
            self.forward_kwargs,
            num_warmups=0,
            num_samples=1,
            ignore_layers=self.ignore_layers,
        )["flops"]


class TorchMacs(ModelEvaluator):
    """Total number of MACs."""

    def __init__(
        self,
        forward_args: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        forward_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ignore_layers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            forward_args: `model.forward()` arguments used for profilling.
            forward_kwargs: `model.forward()` keyword arguments used for profilling.
            ignore_layers: List of layer names that should be ignored during profiling.

        """

        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, budget: Optional[float] = None) -> float:
        return profile(
            model.arch,
            self.forward_args,
            self.forward_kwargs,
            num_warmups=0,
            num_samples=1,
            ignore_layers=self.ignore_layers,
        )["macs"]


class TorchLatency(ModelEvaluator):
    """Average/median latency (in seconds) of a PyTorch model using a sample input."""

    def __init__(
        self,
        forward_args: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        forward_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        num_warmups: Optional[int] = 1,
        num_samples: Optional[int] = 1,
        use_cuda: Optional[bool] = False,
        use_median: Optional[bool] = False,
        ignore_layers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            forward_args: `model.forward()` arguments used for profilling.
            forward_kwargs: `model.forward()` keyword arguments used for profilling.
            num_warmups: Number of warmup runs before profilling.
            num_samples: Number of runs after warmup.
            use_cuda: Whether to use CUDA instead of CPU.
            use_median: Whether to use median instead of mean to average memory and latency.
            ignore_layers: List of layer names that should be ignored during profiling.

        """

        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.ignore_layers = ignore_layers
        self.num_warmups = num_warmups
        self.num_samples = num_samples
        self.use_cuda = use_cuda
        self.use_median = use_median
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, budget: Optional[float] = None) -> float:
        return profile(
            model.arch,
            self.forward_args,
            self.forward_kwargs,
            num_warmups=self.num_warmups,
            num_samples=self.num_samples,
            use_cuda=self.use_cuda,
            use_median=self.use_median,
            ignore_layers=self.ignore_layers,
        )["latency"]


class TorchPeakCudaMemory(ModelEvaluator):
    """Measures CUDA peak memory (in bytes) of a PyTorch model using a sample input."""

    def __init__(
        self,
        forward_args: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        forward_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        num_warmups: Optional[int] = 1,
        num_samples: Optional[int] = 1,
        use_median: Optional[bool] = False,
        ignore_layers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            forward_args: `model.forward()` arguments used for profilling.
            forward_kwargs: `model.forward()` keyword arguments used for profilling.
            num_warmups: Number of warmup runs before profilling.
            num_samples: Number of runs after warmup.
            use_median: Whether to use median instead of mean to average memory and latency.
            ignore_layers: List of layer names that should be ignored during profiling.

        """

        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.ignore_layers = ignore_layers
        self.num_warmups = num_warmups
        self.num_samples = num_samples
        self.use_median = use_median
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, budget: Optional[float] = None) -> float:
        return profile(
            model.arch,
            self.forward_args,
            self.forward_kwargs,
            num_warmups=self.num_warmups,
            num_samples=self.num_samples,
            use_cuda=True,
            use_median=self.use_median,
            ignore_layers=self.ignore_layers,
        )["peak_memory"]


class TorchPeakCpuMemory(ModelEvaluator):
    """Measures CPU peak memory (in bytes) of a PyTorch model using a sample input."""

    def __init__(
        self,
        forward_args: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        forward_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize the evaluator.

        Args:
            forward_args: `model.forward()` arguments used for profilling.
            forward_kwargs: `model.forward()` keyword arguments used for profilling.

        """

        forward_args = forward_args if forward_args is not None else []
        self.forward_args = [forward_args] if isinstance(forward_args, torch.Tensor) else forward_args
        self.forward_kwargs = forward_kwargs or {}

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, budget: Optional[float] = None):
        model.arch.to("cpu")
        forward_args = tuple([arg.to("cpu") for arg in self.forward_args])
        forward_kwargs = {key: value.to("cpu") for key, value in self.forward_kwargs.items()}

        is_training = model.arch.training
        model.arch.eval()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                model.arch(*forward_args, **forward_kwargs)

        event_list = prof.key_averages()
        peak_memory = max(event.cpu_memory_usage for event in event_list)

        if is_training:
            model.arch.train()

        return peak_memory
