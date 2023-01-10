# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Transformer-Flex memory-related objectives."""

import copy
import os
from typing import Any, Dict, Optional

import torch
from overrides import overrides

from archai.discrete_search import ArchaiModel, DatasetProvider, Objective
from archai.nlp.onnx.export import export_to_onnx
from archai.nlp.onnx.export_utils import prepare_model_for_onnx
from archai.nlp.onnx.optimization import optimize_onnx
from archai.nlp.search_spaces.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


class TransformerFlexOnnxMemory(Objective):
    """Implement a Transformer-Flex ONNX memory objective."""

    higher_is_better: bool = False

    def __init__(
        self,
        search_space: TransformerFlexSearchSpace,
        use_past: Optional[bool] = True,
        share_weights: Optional[bool] = True,
        opset: Optional[int] = 11,
        only_ort: Optional[bool] = False,
    ) -> None:
        """Initialize the `TransformerFlexOnnxMemory` instance.

        Args:
            search_space: The search space to use for loading the model.
            use_past: Whether to include past key/values in the model.
            share_weights: Whether to share the embedding and softmax weights.
            opset: Set of operations to use with ONNX.
            only_ort: Whether to only apply ORT optimization.

        """

        assert search_space.arch_type in ["codegen", "gpt2", "gpt2-flex"]
        self.search_space = search_space

        # Benchmark settings
        self.use_past = use_past
        self.share_weights = share_weights
        self.opset = opset
        self.only_ort = only_ort

    def _load_and_prepare(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Load and prepare a model for ONNX conversion.

        Args:
            config: The configuration to use for loading the model.

        Returns:
            The prepared model, ready for ONNX conversion.

        """

        config = copy.deepcopy(config)
        if self.use_past:
            config["use_cache"] = True

        model = self.search_space._load_model_from_config(config)

        return prepare_model_for_onnx(model, self.search_space.arch_type)

    @overrides
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        model = self._load_and_prepare(arch.metadata["config"])

        # There is a bug for Python < 3.10 when using TemporaryFile with Windows,
        # thus, we opted to manually save and remove the temporary file
        tmp_path = "tmp.onnx"

        onnx_config = export_to_onnx(
            model,
            tmp_path,
            task="causal-lm",
            use_past=self.use_past,
            share_weights=self.share_weights,
            opset=self.opset,
        )
        opt_tmp_path = optimize_onnx(tmp_path, onnx_config, opt_level=0, only_ort=self.only_ort)

        memory = os.path.getsize(opt_tmp_path) / (1024**2)

        try:
            os.remove(tmp_path)
            os.remove(opt_tmp_path)
        except FileNotFoundError:
            pass

        return memory
