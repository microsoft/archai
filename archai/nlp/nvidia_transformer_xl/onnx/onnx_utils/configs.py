# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import Any, Dict

import torch


class OnnxConfig:
    """Provides a foundation for creating ONNX-export configuration instances.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        self.config = model_config

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.

        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (1, 32)),
        }

    @property
    def inputs(self) -> None:
        """Defines the inputs and their shapes to be used when exporting to ONNX.

        """

        raise NotImplementedError

    @property
    def outputs(self) -> None:
        """Defines the outputs and their shapes to be used when exporting to ONNX.
        
        """

        raise NotImplementedError


class MemTransformerLMOnnxConfig(OnnxConfig):
    """Provides an ONNX-export configuration for MemTransformerLM.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        super().__init__(model_config)

        if self.config['attn_type'] == 0:
            self.config['past_key_values'] = 3
        else:
            self.config['past_key_values'] = 2

        self.config['model_type'] = 'transfo-xl'

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.
        
        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (1, 32)),
            'past_key_values': tuple([torch.zeros(self.config['past_key_values'], 1, self.config['n_head'], 32, self.config['d_head']) for _ in range(self.config['n_layer'])])
        }

    @property
    def inputs(self) -> OrderedDict:
        """Defines the inputs and their shapes to be used when exporting to ONNX.
        
        """

        pasts = [(f'past_{i}', {0: str(self.config['past_key_values']), 1: 'batch_size', 2: str(self.config['n_head']), 3: 'past_seq_len', 4: str(self.config['d_head'])}) for i in range(self.config['n_layer'])]
        return OrderedDict([('input_ids', {0: 'batch_size', 1: 'seq_len'})] + pasts)

    @property
    def outputs(self) -> OrderedDict:
        """Defines the outputs and their shapes to be used when exporting to ONNX.
        
        """

        presents = [(f'present_{i}', {0: str(self.config['past_key_values']), 1: 'batch_size', 2: str(self.config['n_head']), 3: 'total_seq_len', 4: str(self.config['d_head'])}) for i in range(self.config['n_layer'])]
        return OrderedDict([('probs', {0: 'batch_size', 1: str(self.config['n_token'])})] + presents)
