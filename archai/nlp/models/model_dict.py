# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dictionary and enumerators that allows the implementation
and usage of Transformer-based models.
"""

from enum import Enum
from typing import Dict


class ModelClassType(Enum):
    """An enumerator that defines the type of available classes to be loaded.

    """

    # Types of classes
    MODEL = 0
    CONFIG = 1
    ONNX_CONFIG = 2
    ONNX_MODEL = 3


class ModelDict(Dict):
    """Dictionary that defines the type of available models to be loaded.

    The order of classes must be asserted to the same defined by ModelClassType.

    """

    # Huggingface's Open AI GPT-2
    HF_GPT2 = ('HfGPT2', 'HfGPT2Config', 'HfGPT2OnnxConfig', 'HfGPT2OnnxModel')

    # Huggingface's Transformer-XL
    HF_TRANSFO_XL = ('HfTransfoXL', 'HfTransfoXLConfig', 'HfTransfoXLOnnxConfig', 'HfTransfoXLOnnxModel')

    # NVIDIA's Memory Transfomer (Transformer-XL)
    MEM_TRANSFORMER = ('MemTransformerLM', 'MemTransformerLMConfig', 'MemTransformerLMOnnxConfig', 'MemTransformerLMOnnxModel')
