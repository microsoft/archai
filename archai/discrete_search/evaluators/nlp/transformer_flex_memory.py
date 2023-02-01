# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import pathlib
import shutil
from typing import Any, Dict, Optional

import torch
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)
from archai.onnx.export import export_to_onnx
from archai.onnx.export_utils import prepare_model_for_onnx
from archai.onnx.optimization import optimize_onnx

TMP_FOLDER = pathlib.Path("tmp")


class TransformerFlexOnnxMemory(ModelEvaluator):
    """Measure the memory usage of models from the Transformer-Flex search space."""

    def __init__(
        self,
        search_space: TransformerFlexSearchSpace,
        use_past: Optional[bool] = True,
        validate: Optional[bool] = True,
        share_weights: Optional[bool] = True,
        opset: Optional[int] = 11,
        optimize: Optional[bool] = True,
        only_ort: Optional[bool] = False,
    ) -> None:
        """Initialize the evaluator.

        Args:
            search_space: The search space to use for loading the model.
            use_past: Whether to include past key/values in the model.
            validate: Whether to validate the exported model.
            share_weights: Whether to share the embedding and softmax weights.
            opset: Set of operations to use with ONNX.
            optimize: Whether to optimize the ONNX model.
            only_ort: Whether to only apply ORT optimization.

        """

        assert search_space.arch_type in ["codegen", "gpt2", "gpt2-flex"]
        self.search_space = search_space

        # Benchmark settings
        self.use_past = use_past
        self.validate = validate
        self.share_weights = share_weights
        self.opset = opset
        self.optimize = optimize
        self.only_ort = only_ort

    def _load_and_prepare(self, config: Dict[str, Any]) -> torch.nn.Module:
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
        TMP_FOLDER.mkdir(parents=True, exist_ok=True)
        onnx_path = TMP_FOLDER / "model.onnx"

        onnx_config = export_to_onnx(
            model,
            onnx_path.as_posix(),
            task="causal-lm",
            use_past=self.use_past,
            validate=self.validate,
            share_weights=self.share_weights,
            opset=self.opset,
        )

        if self.optimize:
            onnx_path = optimize_onnx(onnx_path.as_posix(), onnx_config, opt_level=0, only_ort=self.only_ort)

        memory = pathlib.Path(onnx_path).stat().st_size / (1024**2)

        shutil.rmtree(TMP_FOLDER)

        return memory
