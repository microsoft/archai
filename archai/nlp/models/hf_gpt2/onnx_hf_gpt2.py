# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2 for ONNX.
"""

from typing import Any, Dict

import torch
from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel as HfGPT2OnnxModel

from archai.nlp.models.config_base import OnnxConfig


class HfGPT2OnnxConfig(OnnxConfig):
<<<<<<< HEAD
    """Provides an ONNX-export configuration for HfGPT2.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        super().__init__(model_config)
=======
    """Huggingface's Open AI GPT-2 ONNX-based configuration.

    """
>>>>>>> 0a1d1a35 (chore(hf_gpt2): Re-structures hf_gpt2-related files.)

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initializes the class by setting missing keys on incoming
            model's configuration.

        Args:
            model_config: Configuration of the model that will be exported.

        """

        model_config['past_key_values'] = 2
        model_config['model_type'] = 'gpt2'

        super().__init__(model_config)

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.

        Returns:
            (Dict[str, Any]): Mockups used to export with ONNX.
        
        """

        return {
            'input_ids': torch.randint(0, self.config.n_token, (self.batch_size, self.seq_len)),
            'past_key_values': tuple([torch.zeros(self.config.past_key_values, self.batch_size, self.config.n_head, self.seq_len, self.config.d_head) for _ in range(self.config.n_layer)])
        }
