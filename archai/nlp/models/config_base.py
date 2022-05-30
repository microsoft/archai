# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Configuration-related base classes, such as default, search and ONNX-based.
"""

from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, List, Union

import torch

from transformers import PretrainedConfig


class Config(PretrainedConfig):
    """Base configuration class, used to define some common attributes
        and shared methods for loading and saving configurations.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the class by verifying whether keyword arguments
            are valid and setting them as attributes.

        """

        super().__init__(**kwargs)

        # Re-assures that model_type overwrites interal attribute from base class
        if 'model_type' in kwargs:
            self.__class__.model_type = kwargs['model_type']

    def _map_to_list(self,
                     variable: Union[int, float, List[Union[int, float]]],
                     size: int) -> List[Union[int, float]]:
        if isinstance(variable, list):
            size_diff = size - len(variable)

            if size_diff < 0:
                return variable[:size]
            if size_diff == 0:
                return variable
            if size_diff > 0:
                return variable + [variable[0]] * size_diff

        return [variable] * size

    def to_dict(self) -> Dict[str, Any]:
        config_dict = super().to_dict()
        
        for key, mapped_key in self.attribute_map.items():
            config_dict[key] = config_dict[mapped_key]

        return config_dict


class SearchConfigParameter:
    """Base search configuration parameter class, used to define whether
        it should be different for each layer and its possible values.

    """

    def __init__(self,
                 per_layer: Optional[bool] = False,
                 value: Optional[List[Any]] = None) -> None:
        """Initializes the class by setting basic attributes
            of a search config parameter.

        Args:
            per_layer: Whether parameter should be different for each layer.
            value: Possible parameter values.

        """

        self.per_layer = per_layer
        self.value = value if value is not None else [1]

    def to_dict(self) -> Dict[str, Any]:
        search_config_parameter_dict = {}

        for key, value in self.__dict__.items():
            search_config_parameter_dict[key] = value

        return search_config_parameter_dict


class SearchConfig:
    """Base search configuration class, used to define possible
        hyperparameters that can used during search.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the class by setting keywords as attributes.

        Note that the name of the keyword has to match the name of the parameter
            that will be used during search.

        """

        for key, value in kwargs.items():
            assert isinstance(value, SearchConfigParameter)
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        search_config_dict = {}

        for key, value in self.__dict__.items():
            search_config_dict[key] = value.to_dict()

        return search_config_dict


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
