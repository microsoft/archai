from functools import partial
from typing import Tuple, List, Optional, Dict

import torch
from overrides import overrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.evaluators.torch_profiler_utils import profile


class TorchNumParameters(ModelEvaluator):
    def __init__(self, exclude_cls: Optional[List[torch.nn.Module]] = None, trainable_only: Optional[bool] = True) -> None:
        """Counts the total number of trainable parameters

        Args:
            exclude_cls (Optional[List[torch.nn.Module]], optional): List of PyTorch module classes
                to exclude from parameter counting. Defaults to None.

            trainable_only (Optional[bool], optional): A flag indicating whether only trainable parameters
                should be counted. Defaults to True.
        """
        self.exclude_cls = exclude_cls
        self.trainable_only = trainable_only

    @overrides
    def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        total_params = sum(
            param.numel() for param in model.arch.parameters() if not self.trainable_only or param.requires_grad
        )

        exclude_params = 0 if self.exclude_cls is None else sum(
            sum(param.numel() for param in module.parameters())
            for module in model.arch.modules()
            if isinstance(module, tuple(self.exclude_cls))
        )

        return total_params - exclude_params


class TorchFlops(ModelEvaluator):
    def __init__(self, sample_args: Optional[Tuple[torch.Tensor]] = None,
                 sample_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                 ignore_layers: Optional[List[str]] = None):
        """Calculates FLOPs of a PyTorch model using a sample input.

        Args:
            sample_args (Optional[Tuple[torch.Tensor]], optional): `model.forward` arguments used
                for profilling. Defaults to None.
            
            sample_kwargs (Optional[Dict[str, torch.Tensor]], optional): `model.forward` keyword
                arguments used for profilling. Defaults to None.

            ignore_layers (Optional[List[str]], optional): List of layer names that should be ignored during 
                the stat calculation. Defaults to None
        """
        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, 
                budget: Optional[float] = None) -> float:
        return profile(
            model.arch, self.sample_args, self.sample_kwargs, num_warmups=0, num_samples=1,
            ignore_layers=self.ignore_layers
        )['flops']


class TorchMacs(ModelEvaluator):
    def __init__(self, sample_args: Optional[Tuple[torch.Tensor]] = None,
                 sample_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                 ignore_layers: Optional[List[str]] = None):
        """Calculates MACs of a PyTorch model using a sample input.

        Args:
            sample_args (Optional[Tuple[torch.Tensor]], optional): `model.forward` arguments used
                for profilling. Defaults to None.
            
            sample_kwargs (Optional[Dict[str, torch.Tensor]], optional): `model.forward` keyword
                arguments used for profilling. Defaults to None.

            ignore_layers (Optional[List[str]], optional): List of layer names that should be ignored during 
                the stat calculation. Defaults to None
        """
        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, 
                budget: Optional[float] = None) -> float:
        return profile(
            model.arch, self.sample_args, self.sample_kwargs, num_warmups=0, num_samples=1,
            ignore_layers=self.ignore_layers
        )['macs']


class TorchLatency(ModelEvaluator):
    def __init__(self, sample_args: Optional[Tuple[torch.Tensor]] = None,
                 sample_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                 num_warmups: Optional[int] = 1,
                 num_samples: Optional[int] = 1,
                 use_median: Optional[bool] = False,
                 ignore_layers: Optional[List[str]] = None):
        """Calculates the average/median latency (in seconds) of a PyTorch model using a sample input.

        Args:
            sample_args (Optional[Tuple[torch.Tensor]], optional): `model.forward` arguments used
                for profilling. Defaults to None.
            
            sample_kwargs (Optional[Dict[str, torch.Tensor]], optional): `model.forward` keyword
                arguments used for profilling. Defaults to None.

            num_warmups (int, optional): Number of warmup runs before profilling. Defaults to 1.

            num_samples (int, optional): Number of runs after warmup. Defaults to 1

            use_median (bool, optional): Whether to use median instead of mean to average memory and latency. 
                Defaults to False.

            ignore_layers (Optional[List[str]], optional): List of layer names that should be ignored during 
                the stat calculation. Defaults to None
        """
        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs
        self.num_warmups = num_warmups
        self.num_samples = num_samples
        self.use_median = use_median
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, 
                budget: Optional[float] = None) -> float:
        return profile(
            model.arch, self.sample_args, self.sample_kwargs,
            num_warmups=self.num_warmups, num_samples=self.num_samples,
            use_median=self.use_median, ignore_layers=self.ignore_layers
        )['latency']


class TorchCudaPeakMemory(ModelEvaluator):
    def __init__(self, sample_args: Optional[Tuple[torch.Tensor]] = None,
                 sample_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                 num_warmups: Optional[int] = 1,
                 num_samples: Optional[int] = 1,
                 use_median: Optional[bool] = False,
                 ignore_layers: Optional[List[str]] = None):
        """Calculates the average/median CUDA peak memory (in bytes) of a PyTorch model using a sample input.
        All inputs passed must be on the same CUDA device as the model.

        Args:
            sample_args (Optional[Tuple[torch.Tensor]], optional): `model.forward` arguments used
                for profilling. Defaults to None.
            
            sample_kwargs (Optional[Dict[str, torch.Tensor]], optional): `model.forward` keyword
                arguments used for profilling. Defaults to None.

            num_warmups (int, optional): Number of warmup runs before profilling. Defaults to 1.

            num_samples (int, optional): Number of runs after warmup. Defaults to 1

            use_median (bool, optional): Whether to use median instead of mean to average memory and latency. 
                Defaults to False.

            ignore_layers (Optional[List[str]], optional): List of layer names that should be ignored during 
                the stat calculation. Defaults to None
        """
        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs
        self.num_warmups = num_warmups
        self.num_samples = num_samples
        self.use_median = use_median
        self.ignore_layers = ignore_layers

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, 
                budget: Optional[float] = None) -> float:
        return profile(
            model.arch, self.sample_args, self.sample_kwargs,
            num_warmups=self.num_warmups, num_samples=self.num_samples,
            use_median=self.use_median, ignore_layers=self.ignore_layers
        )['peak_memory']
