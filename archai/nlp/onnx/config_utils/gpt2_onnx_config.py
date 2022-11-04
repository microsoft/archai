# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2 ONNX configuration.
"""

from typing import Any, Dict

from archai.nlp.onnx.config_utils.onnx_config_base import OnnxConfigWithPast


class HfGPT2OnnxConfig(OnnxConfigWithPast):
    """Huggingface's Open AI GPT-2 ONNX-based configuration.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initializes the class by setting missing keys on incoming
            model's configuration.
        Args:
            model_config: Configuration of the model that will be exported.
        """

        super().__init__(model_config, model_type='gpt2', past_key_values=2)
