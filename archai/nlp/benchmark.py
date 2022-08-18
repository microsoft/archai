# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Customizable benchmark utilities from huggingface/transformers.
"""

from transformers.benchmark.benchmark import PyTorchBenchmark
from transformers.benchmark.benchmark_args import PyTorchBenchmarkArguments


class ArchaiBenchmarkArguments(PyTorchBenchmarkArguments):
    """Inherits from PyTorchBenchmarkArguments and defines certain keywords."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        # Overrides default keywords
        kwargs["inference"] = True
        kwargs["memory"] = True
        kwargs["training"] = False
        kwargs["only_pretrain_model"] = True
        kwargs["env_print"] = True

        super().__init__(*args, **kwargs)


class ArchaiBenchmark(PyTorchBenchmark):
    """Inherits from PyTorchBenchmark and allows to be customized."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        super().__init__(*args, **kwargs)
