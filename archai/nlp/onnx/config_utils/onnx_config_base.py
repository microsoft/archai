# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-based configuration.
"""

from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional

import torch


class OnnxConfig:
    """Base ONNX configuration class, used to define mockups,
        inputs and outputs dictionaries that should be used during export.

    """

    def __init__(self,
                 model_config: Dict[str, Any],
                 model_type: Optional[str] = '',
                 batch_size: Optional[int] = 1,
                 seq_len: Optional[int] = 32) -> None:
        """Initializes the class by creating a configuration object and setting
            common-shared attributes.

        Args:
            model_config: Configuration of model that will be exported.
            model_type: Type of model that will be exported.
            batch_size: Batch size of dummy inputs.
            seq_len: Sequence length of dummy inputs.
            
        """

        self.config = Config(**model_config, model_type=model_type)
        self.batch_size = batch_size
        self.seq_len = seq_len

    @property
    def mockups(self) -> Mapping[str, torch.Tensor]:
        input_ids = torch.randint(0, self.config.n_token, (self.batch_size, self.seq_len))
        
        return OrderedDict({'input_ids': input_ids})

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        input_ids = [('input_ids', {0: 'batch_size', 1: 'seq_len'})]
        
        return OrderedDict(input_ids)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        probs = [('probs', {0: 'batch_size'})]

        return OrderedDict(probs)

    def to_dict(self) -> Dict[str, Any]:
        onnx_config_parameter_dict = {}

        for key, value in self.__dict__.items():
            onnx_config_parameter_dict[key] = value

        return onnx_config_parameter_dict


class OnnxConfigWithPast(OnnxConfig):
    """ONNX configuration class that allows usage of past states (`past_key_values`).

    """

    def __init__(self,
                 model_config: Dict[str, Any],
                 model_type: Optional[str] = '',
                 past_key_values: Optional[int] = 2,
                 batch_size: Optional[int] = 1,
                 seq_len: Optional[int] = 32) -> None:
        """Initializes the class by creating a configuration object and setting
            common-shared attributes.

        Args:
            model_config: Configuration of model that will be exported.
            model_type: Type of model that will be exported.
            past_key_values: Tensors (key, value, etc) that are saved in past states (defaults to `2`).
            batch_size: Batch size of dummy inputs.
            seq_len: Sequence length of dummy inputs.
            
        """

        model_config['past_key_values'] = past_key_values

        super().__init__(model_config,
                         model_type=model_type,
                         batch_size=batch_size,
                         seq_len=seq_len)

    @property
    def mockups(self) -> Mapping[str, torch.Tensor]:
        input_ids = torch.randint(0, self.config.n_token, (self.batch_size, self.seq_len))

        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        past_key_values =  tuple([torch.zeros(self.config.past_key_values, self.batch_size, self.config.n_head, self.seq_len, self.config.d_head) for _ in range(self.config.n_layer)])

        return OrderedDict({'input_ids': input_ids, 'past_key_values': past_key_values})

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        input_ids = [('input_ids', {0: 'batch_size', 1: 'seq_len'})]

        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        past_key_values = [(f'past_{i}', {1: 'batch_size', 3: 'past_seq_len'}) for i in range(self.config.n_layer)]

        return OrderedDict(input_ids + past_key_values)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        probs = [('probs', {0: 'batch_size'})]

        # Shape of present states (past states when outputting)
        # [past_key_values, batch_size, n_head, total_seq_len, d_head]
        # Note that total_seq_len is seq_len + past_seq_len
        present_key_values = [(f'present_{i}', {1: 'batch_size', 3: 'total_seq_len'}) for i in range(self.config.n_layer)]

        return OrderedDict(probs + present_key_values)