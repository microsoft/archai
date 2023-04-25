# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List, Optional, Tuple, Union
from overrides import overrides
from archai.discrete_search.evaluators import AvgOnnxLatency
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.common.store import ArchaiStore


class AvgOnnxLatencyEvaluator(AvgOnnxLatency):
    """Evaluate the average ONNX Latency (in seconds) of an architecture and store the result.
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
        device: Optional[str] = 'cpu',
        inf_session_kwargs: Optional[Dict[str, Any]] = None,
        store: ArchaiStore = None,
        metric_key: str = 'onnx_latency'
    ) -> None:
        super(AvgOnnxLatencyEvaluator, self).__init__(
            input_shape,
            num_trials,
            input_dtype,
            rand_range,
            export_kwargs,
            device,
            inf_session_kwargs)
        self.store = store
        self.metric_key = metric_key
        self.iteration = 1

    @overrides
    def evaluate(self, model: ArchaiModel, budget: Optional[float] = None) -> float:
        archid = f'id_{model.archid}'
        if self.store is not None:
            e = self.store.get_status(archid)
            if 'iteration' not in e or e['iteration'] != self.iteration:
                e['iteration'] = self.iteration
                self.store.merge_status_entity(e)
            if self.metric_key in e:
                # Use the cached value and skip the more expensive Onnx evaluation.
                # This also ensures maximum re-use of previous training jobs.
                return e[self.metric_key]

        result = super(AvgOnnxLatencyEvaluator, self).evaluate(model, budget)
        if self.store is not None:
            e = self.store.get_status(archid)
            e['status'] = 'complete'
            e[self.metric_key] = result
            self.store.merge_status_entity(e)
        return result

    def on_start_iteration(self, iteration: int):
        self.iteration = iteration + 1
