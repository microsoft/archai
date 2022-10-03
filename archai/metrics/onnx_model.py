from typing import Tuple, Union, List, Dict, Optional
import io

import torch
from overrides import overrides
import onnxruntime as rt

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric
from archai.common.timing import MeasureBlockTime


class AvgOnnxLatencyMetric(BaseMetric):
    def __init__(self, input_shape: Union[Tuple, List[Tuple]], num_trials: int = 1,
                 export_kwargs: Optional[Dict] = None, inf_session_kwargs: Optional[Dict] = None):
        """Average ONNX Latency Metric (in seconds).

        Args:
            input_shape (Union[Tuple, List[Tuple]]): Model Input shape or list of model input shapes.
            num_trials (int, optional): Number of trials. Defaults to 1.
            export_kwargs (Optional[Dict], optional): Optional dictionary of key-value args passed to
                `torch.onnx.export`. Defaults to None.
            inf_session_kwargs (Optional[Dict], optional): Optional dictionary of key-value args 
                passed to `onnxruntime.InferenceSession()`. Defaults to None.
        """
        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape            
        
        self.sample_input = tuple([torch.rand(*input_shape) for input_shape in input_shapes])
        self.num_trials = num_trials
        self.export_kwargs = export_kwargs or dict()
        self.inf_session_kwargs = inf_session_kwargs or dict()

    @overrides
    def compute(self, model: ArchWithMetaData, dataset_provider: DatasetProvider,
                budget: Optional[float] = None) -> float:
        model.arch.to('cpu')

        # Exports model to ONNX
        exported_model_buffer = io.BytesIO()
        torch.onnx.export(
            model.arch, self.sample_input, exported_model_buffer,
            input_names=[f'input_{i}' for i in range(len(self.sample_input))],
            **self.export_kwargs
        )
        
        exported_model_buffer.seek(0)

        # Benchmarks ONNX model
        onnx_session = rt.InferenceSession(exported_model_buffer.read(), **self.inf_session_kwargs)
        sample_input = {f'input_{i}': inp.numpy() for i, inp in enumerate(self.sample_input)}
        inf_times = []

        for _ in range(self.num_trials):
            with MeasureBlockTime('onnx_inference') as t:
                onnx_session.run(None, input_feed=sample_input)
            inf_times.append(t.elapsed)

        return sum(inf_times) / self.num_trials
