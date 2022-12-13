# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Transformer-Flex latency-related objectives.
"""

import copy
import tempfile
import timeit
from typing import Any, Dict, Optional

import numpy as np
import torch
from onnxruntime import InferenceSession
from overrides import overrides

from archai.discrete_search import ArchaiModel, DatasetProvider, Objective
from archai.nlp.onnx.config_utils.onnx_config_base import OnnxConfig
from archai.nlp.onnx.export import export_to_onnx
from archai.nlp.onnx.export_utils import prepare_model_for_onnx
from archai.nlp.onnx.onnx_loader import load_from_onnx
from archai.nlp.onnx.optimization import optimize_onnx
from archai.nlp.search_spaces.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


class TransformerFlexOnnxLatency(Objective):
    """Implements a Transformer-Flex ONNX latency objective."""

    higher_is_better: bool = False

    def __init__(
        self,
        search_space: TransformerFlexSearchSpace,
        batch_size: Optional[int] = 1,
        seq_len: Optional[int] = 192,
        past_seq_len: Optional[int] = 0,
        n_trials: Optional[int] = 1,
        use_median: Optional[bool] = False,
    ) -> None:
        assert search_space.arch_type in ["gpt2", "gpt2-flex"]
        self.search_space = search_space

        # Benchmark settings
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.past_seq_len = past_seq_len
        self.n_trials = n_trials
        self.use_median = use_median

    def _load_and_prepare(self, config: Dict[str, Any]) -> torch.nn.Module:
        config = copy.deepcopy(config)
        config["use_cache"] = True

        model = self.search_space._load_model_from_config(config)

        return prepare_model_for_onnx(model, self.search_space.arch_type)

    def _benchmark_model(self, session: InferenceSession, model_config: OnnxConfig) -> float:
        inputs = model_config.generate_dummy_inputs(self.batch_size, self.seq_len, self.past_seq_len)
        past_inputs = inputs.pop("past_key_values")

        for i, past in enumerate(past_inputs):
            inputs[f"past_{i}"] = past

        timer = timeit.Timer(
            stmt="onnx_model_session(None, inputs)",
            globals={"inputs": {k: v.numpy() for k, v in inputs.items()}, "onnx_model_session": session.run},
        )

        # Performs a quick warmup prior to the calculation
        _ = timer.timeit(number=max(int(self.n_trials // 100), 2))

        # Calculates proper set of times (instead of sum)
        runner = timer.repeat(repeat=self.n_trials, number=self.n_trials)
        runner = [r / self.n_trials for r in runner]

        return float(np.median(runner) if self.use_median else np.mean(runner))

    @overrides
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        model = self._load_and_prepare(arch.metadata["config"])

        with tempfile.NamedTemporaryFile() as tmp:
            tmp_path = tmp.name

            onnx_config = export_to_onnx(model, tmp_path, task="causal-lm", use_past=True, share_weights=True, opset=11)
            opt_tmp_path = optimize_onnx(tmp_path, onnx_config, opt_level=0)

            session = load_from_onnx(opt_tmp_path)

            return self._benchmark_model(session, onnx_config)
