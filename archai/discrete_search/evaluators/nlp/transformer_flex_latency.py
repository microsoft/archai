# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import pathlib
import shutil
import timeit
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from onnxruntime import InferenceSession
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)
from archai.onnx.config_utils.onnx_config_base import OnnxConfig
from archai.onnx.export import export_to_onnx
from archai.onnx.export_utils import prepare_model_for_onnx
from archai.onnx.onnx_loader import load_from_onnx
from archai.onnx.optimization import optimize_onnx

TMP_FOLDER = pathlib.Path("tmp")


class TransformerFlexOnnxLatency(ModelEvaluator):
    """Measure the average latency of models from the Transformer-Flex search space."""

    def __init__(
        self,
        search_space: TransformerFlexSearchSpace,
        providers: Optional[List[str]] = None,
        batch_size: Optional[int] = 1,
        seq_len: Optional[int] = 192,
        past_seq_len: Optional[int] = 0,
        n_trials: Optional[int] = 1,
        use_median: Optional[bool] = False,
        use_past: Optional[bool] = True,
        validate: Optional[bool] = True,
        share_weights: Optional[bool] = True,
        opset: Optional[int] = 11,
        optimize: Optional[bool] = True,
        only_ort: Optional[bool] = False,
    ) -> None:
        """Initialize the evaluator.

        This evaluator supports measuring in different ONNX Runtime providers. For measuring on
        GPUs, use `providers=["CUDAExecutionProvider"]` and make sure that `onnxruntime-gpu`
        package is installed.

        Args:
            search_space: The search space to use for loading the model.
            providers: The list of ORT providers to use for benchmarking.
            batch_size: The batch size to use when benchmarking the model.
            seq_len: The sequence length to use when benchmarking the model.
            past_seq_len: The past sequence length to use when benchmarking the model.
            n_trials: The number of trials to use when benchmarking the model.
            use_median: Whether to use the median or the mean of the measured
                times as the result.
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
        self.providers = providers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.past_seq_len = past_seq_len
        self.n_trials = n_trials
        self.use_median = use_median
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

    def _benchmark_model(self, session: InferenceSession, model_config: OnnxConfig) -> float:
        inputs = model_config.generate_dummy_inputs(self.batch_size, self.seq_len, self.past_seq_len)

        if self.use_past:
            past_inputs = inputs.pop("past_key_values")
            for i, past in enumerate(past_inputs):
                inputs[f"past_{i}"] = past

        timer = timeit.Timer(
            stmt="onnx_model_session(None, inputs)",
            globals={"inputs": {k: v.numpy() for k, v in inputs.items()}, "onnx_model_session": session.run},
        )

        # Perform a quick warmup prior to the calculation
        _ = timer.timeit(number=max(int(self.n_trials // 100), 2))

        # Calculate proper set of times (instead of sum)
        runner = timer.repeat(repeat=self.n_trials, number=self.n_trials)
        runner = [r / self.n_trials for r in runner]

        return float(np.median(runner) if self.use_median else np.mean(runner))

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

        session = load_from_onnx(onnx_path, providers=self.providers)
        latency = self._benchmark_model(session, onnx_config)

        shutil.rmtree(TMP_FOLDER)

        return latency
