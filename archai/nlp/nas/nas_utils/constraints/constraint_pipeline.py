# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constraints pipelines used throughout the search procedure.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from archai.nlp.nas.nas_utils.constraints.onnx_constraints import (measure_onnx_disk_memory,
                                                                   measure_onnx_inference_latency,
                                                                   measure_onnx_parameters)
from archai.nlp.nas.nas_utils.constraints.torch_constraints import (measure_torch_char_accept_rate,
                                                                    measure_torch_inference_latency,
                                                                    measure_torch_parameters,
                                                                    measure_torch_peak_memory,
                                                                    measure_torch_val_ppl)

# Latency upper bound on different device targets
# Any model with more latency than this will be removed from consideration during search
DEVICE_LATENCY_CONSTRAINT = {
    'XeonE5-2690': 10.0,
    'corei7': 10.0,
    'corei5': 10.0,
    'D3_V2': 10.0,
}


class ConstraintPipeline:
    """Defines a constraint pipeline that enables multi-typing (PyTorch and ONNX).

    """

    def __init__(self,
                 use_quantization: bool,
                 use_median: bool,
                 batch_size: int,
                 seq_len: int,
                 n_threads: int,
                 n_trials: int,
                 device: str) -> None:
        """Overrides initialization method.

        Args:
            use_quantization: Whether measurement should be calculated with quantizated model or not.
            use_median: Whether should use median instead of mean for measurement.
            batch_size: Batch size.
            seq_len: Sequence length.
            n_threads: Number of inference threads.
            n_trials: Number of times to repeat the measurement.
            device: Device where should be measured.

        """

        self.use_quantization = use_quantization
        self.use_median = use_median

        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.n_threads = n_threads
        self.n_trials = n_trials

        self.device = device

    def __call__() -> None:
        """Invokes the built-in call method.

        This method has to be implemented for child classes according to
        their pipeline usage.

        In Archai, we adopt the following:
            - PyTorch: call(model)
            - ONNX: call(model_config)

        Raises:
            Not implemented error.

        """

        raise NotImplementedError


class TorchConstraintPipeline(ConstraintPipeline):
    """Defines a PyTorch-based constraint pipeline.

    """

    def __init__(self,
                 training_strategy: Optional[str] = 'decoder_params',
                 dataset: Optional[str] = 'wt103',
                 scoring_file: Optional[str] = None,
                 vocab_type: Optional[str] = 'word',
                 vocab_size: Optional[int] = 10000,
                 training_max_step: Optional[int] = 100,
                 use_quantization: Optional[bool] = False,
                 use_median: Optional[bool] = False,
                 batch_size: Optional[int] = 1,
                 seq_len: Optional[int] = 192,
                 n_threads: Optional[int] = 1,
                 n_trials: Optional[int] = 10,
                 device: Optional[str] = 'cpu') -> None:
        """Overrides initialization method.

        Args:
            training_strategy: Training strategy (`decoder_params`, `val_ppl` or `char_accept_rate`).
            dataset: Dataset (if not using `decoder_params`).
            scoring_file: Scoring .ljson file (if using `char_accept_rate`).
            vocab_type: Type of vocabulary (if not using `decoder_params`).
            vocab_size: Size of vocabulary (if not using `decoder_params`).
            training_max_step: Maximum training steps (if not using `decoder_params`).
            use_quantization: Whether measurement should be calculated with quantizated model or not.
            use_median: Whether should use median instead of mean for measurement.
            batch_size: Batch size.
            seq_len: Sequence length.
            n_threads: Number of inference threads.
            n_trials: Number of times to repeat the measurement.
            device: Device where should be measured.

        """

        self.training_strategy = training_strategy
        self.dataset = dataset
        self.scoring_file = scoring_file
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size
        self.training_max_step = training_max_step

        super().__init__(use_quantization, use_median, batch_size,
                         seq_len, n_threads, n_trials, device)

    def __call__(self,
                 model: torch.nn.Module,
                 model_config: Dict[str, Any]) -> Tuple[Union[int, float], int, float, float]:
        """Invokes the built-in call method.

        Args:
            model: Model to be used within constraint pipeline.
            model_config: Configuration of model.

        Returns:
            (Tuple[Union[int, float], int, float, float]): Decoder parameters, validation perplexity
                or character accept rate, total parameters, latency and peak memory.

        """

        if self.training_strategy == 'decoder_params':
            # Number of decoder (non-embedding) parameters
            measure_torch_proxy = measure_torch_parameters(model, ['non_embedding'])
        elif self.training_strategy == 'val_ppl':
            # Validation perplexity
            _, measure_torch_proxy = measure_torch_val_ppl(model,
                                                           model_config,
                                                           dataset=self.dataset,
                                                           vocab_type=self.vocab_type,
                                                           vocab_size=self.vocab_size,
                                                           max_step=self.training_max_step)
        elif self.training_strategy == 'char_accept_rate':
            # Text Predict with character acceptance rate
            measure_torch_proxy = measure_torch_char_accept_rate(model,
                                                                 model_config,
                                                                 dataset=self.dataset,
                                                                 scoring_file=self.scoring_file,
                                                                 vocab_type=self.vocab_type,
                                                                 vocab_size=self.vocab_size,
                                                                 max_step=self.training_max_step)
        else:
            raise NotImplementedError(f'training_strategy: {self.training_strategy} has not been implemented yet.')

        return (
            # Proxy (decoder parameters, validation perplexity or character acceptance rate)
            measure_torch_proxy,

            # Number of total parameters
            measure_torch_parameters(model, ['total']),

            # Inference latency
            measure_torch_inference_latency(model,
                                            self.use_quantization,
                                            self.use_median,
                                            self.batch_size,
                                            self.seq_len,
                                            self.n_threads,
                                            self.n_trials,
                                            self.device),

            # Peak memory usage
            measure_torch_peak_memory(model,
                                      self.use_quantization,
                                      self.batch_size,
                                      self.seq_len,
                                      self.n_threads,
                                      self.device)
        )


class ONNXConstraintPipeline(ConstraintPipeline):
    """Defines a ONNX-based constraint pipeline.

    """

    def __init__(self,
                 use_quantization: Optional[bool] = False,
                 use_median: Optional[bool] = False,
                 batch_size: Optional[int] = 1,
                 seq_len: Optional[int] = 192,
                 n_trials: Optional[int] = 10) -> None:
        """Overrides initialization method.

        Args:
            use_quantization: Whether measurement should be calculated with quantizated model or not.
            use_median: Whether should use median instead of mean for measurement.
            batch_size: Batch size.
            seq_len: Sequence length.
            n_trials: Number of times to repeat the measurement.

        """

        super().__init__(use_quantization, use_median, batch_size,
                         seq_len, 1, n_trials, 'cpu')

    def __call__(self, model_type: str, model_config: Dict[str, Any]) -> Tuple[int, int, float, float]:
        """Invokes the built-in call method.

        Args:
            model_type: Type of model.
            model_config: Model's configuration.

        Returns:
            (Tuple[int, int, float, float]): Decoder parameters, total parameters,
                latency and disk memory.
            
        """

        return (
            # Number of decoder (non-embedding) parameters
            measure_onnx_parameters(model_type, model_config, 'non_embedding'),

            # Number of total parameters
            measure_onnx_parameters(model_type, model_config, 'total'),

            # Inference latency
            measure_onnx_inference_latency(model_type,
                                           model_config,
                                           self.use_quantization,
                                           self.use_median,
                                           self.batch_size,
                                           self.seq_len,
                                           self.n_trials),

            # Disk memory
            measure_onnx_disk_memory(model_type,
                                     model_config,
                                     self.use_quantization)
        )
