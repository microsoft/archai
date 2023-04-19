import io
from typing import Dict, List, Optional, Tuple, Union

import os
os.environ["OMP_NUM_THREADS"] = "1"
import statistics
from time import perf_counter

import onnxruntime as rt
import torch

from archai.discrete_search.api.archai_model import ArchaiModel


class AvgOnnxLatency:
    higher_is_better: bool = False

    def __init__(self, input_shape: Union[Tuple, List[Tuple]], num_trials: int = 15, num_input: int = 15,
                 input_dtype: str = 'torch.FloatTensor', rand_range: Tuple[float, float] = (0.0, 1.0),
                 export_kwargs: Optional[Dict] = None, inf_session_kwargs: Optional[Dict] = None):
        """ Measure the average ONNX Latency (in millseconds) of a model

        Args:
            input_shape (Union[Tuple, List[Tuple]]): Model Input shape or list of model input shapes.
            num_trials (int, optional): Number of trials. Defaults to 15.
            num_input (int, optional): Number of input per trial. Defaults to 15.
            input_dtype (str, optional): Data type of input samples.
            rand_range (Tuple[float, float], optional): The min and max range of input samples.
            export_kwargs (Optional[Dict], optional): Optional dictionary of key-value args passed to
                `torch.onnx.export`. Defaults to None.
            inf_session_kwargs (Optional[Dict], optional): Optional dictionary of key-value args 
                passed to `onnxruntime.InferenceSession()`. Defaults to None.
        """
        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape            
        
        rand_min, rand_max = rand_range
        self.sample_input = tuple([
            ((rand_max - rand_min) * torch.rand(*input_shape) + rand_min).type(input_dtype)
            for input_shape in input_shapes
        ])

        self.num_trials = num_trials
        self.num_input_per_trial = num_input
        self.export_kwargs = export_kwargs or dict()
        self.inf_session_kwargs = inf_session_kwargs or dict()

    def evaluate(self, model: ArchaiModel) -> float:
        """Evaluate the model and return the average latency (in milliseconds)"""
        """Args: model (ArchaiModel): Model to evaluate"""
        """Returns: float: Average latency (in milliseconds)"""

        model.arch.to('cpu')
        exported_model_buffer = io.BytesIO()
        torch.onnx.export(
            model.arch, self.sample_input, exported_model_buffer,
            input_names=[f'input_{i}' for i in range(len(self.sample_input))],
            opset_version=11,
            **self.export_kwargs
        )
        exported_model_buffer.seek(0)

        opts = rt.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        onnx_session = rt.InferenceSession(exported_model_buffer.read(), sess_options=opts, **self.inf_session_kwargs)
        sample_input = {f'input_{i}': inp.numpy() for i, inp in enumerate(self.sample_input)}
        inf_time_avg = self.get_time_elapsed (onnx_session, sample_input, num_input = self.num_input_per_trial, num_measures = self.num_trials)

        return inf_time_avg

    
    def get_time_elapsed (self, onnx_session, sample_input, num_input:int = 15, num_measures:int = 15) -> float:
        """Measure the average time elapsed (in milliseconds) for a given model and input for anumber of times
        Args:
            onnx_session (onnxruntime.InferenceSession): ONNX Inference Session
            sample_input (Dict[str, np.ndarray]): Sample input to the model
            num_input (int, optional): Number of input per trial. Defaults to 15.
            num_measures (int, optional): Number of measures. Defaults to 15.
        Returns:
            float: Average time elapsed (in milliseconds)"""

        def meausre_func() : 
            """Measure the time elapsed (in milliseconds) for a given model and input, once
            Returns: float: Time elapsed (in milliseconds)"""

            t0 = perf_counter()
            for _ in range(num_input): 
                onnx_session.run(None, input_feed=sample_input)[0]
            t1 = perf_counter()                
            time_measured = 1e3 * (t1 - t0) / num_input
            return time_measured

        return statistics.mean([meausre_func() for _ in range (num_measures)])