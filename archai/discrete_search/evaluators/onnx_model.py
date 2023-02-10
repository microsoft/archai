# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import onnxruntime as rt
import torch
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.common.timing import MeasureBlockTime
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator


class AvgOnnxLatency(ModelEvaluator):
    """Evaluate the average ONNX Latency (in seconds) of an architecture.

    The latency is measured by running the model on random inputs and averaging the latency over
    `num_trials` trials.

    """

    def __init__(
        self,
        input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        num_trials: Optional[int] = 1,
        input_dtype: Optional[str] = "torch.FloatTensor",
        rand_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        export_kwargs: Optional[Dict[str, Any]] = None,
        inf_session_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            input_shape: Input shape(s) of the model. If a list of shapes is provided,
                the model is assumed to have multiple inputs.
            num_trials: Number of trials to run.
            input_dtype: Data type of the input.
            rand_range: Range of random values to use for the input.
            export_kwargs: Keyword arguments to pass to `torch.onnx.export`.
            inf_session_kwargs: Keyword arguments to pass to `onnxruntime.InferenceSession`.

        """

        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape

        rand_min, rand_max = rand_range
        self.sample_input = tuple(
            [
                ((rand_max - rand_min) * torch.rand(*input_shape) + rand_min).type(input_dtype)
                for input_shape in input_shapes
            ]
        )

        self.input_dtype = input_dtype
        self.rand_range = rand_range
        self.num_trials = num_trials
        self.export_kwargs = export_kwargs or dict()
        self.inf_session_kwargs = inf_session_kwargs or dict()

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, budget: Optional[float] = None) -> float:
        model.arch.to("cpu")

        # Exports model to ONNX
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            torch.onnx.export(
                model.arch,
                self.sample_input,
                tmp_file.name,
                input_names=[f"input_{i}" for i in range(len(self.sample_input))],
                **self.export_kwargs,
            )

            # Benchmarks ONNX model
            onnx_session = rt.InferenceSession(tmp_file.name, **self.inf_session_kwargs)
            sample_input = {f"input_{i}": inp.numpy() for i, inp in enumerate(self.sample_input)}
            inf_times = []

            for _ in range(self.num_trials):
                with MeasureBlockTime("onnx_inference") as t:
                    onnx_session.run(None, input_feed=sample_input)
                inf_times.append(t.elapsed)
        
        # Manually delete tempfile for windows compatibility
        os.remove(tmp_file.name)
        
        return sum(inf_times) / self.num_trials
