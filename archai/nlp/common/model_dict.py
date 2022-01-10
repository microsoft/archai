# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dictionary and enumerators that allows the implementation
and usage of Transformer-based models.
"""

from enum import Enum
from typing import Dict


class ClassType(Enum):
    """An enumerator that defines the type of available classes to be loaded.

    """

    # Types of classes
    MODEL = 0
    ONNX_CONFIG = 1
    ONNX_MODEL = 2


class ModelDict(Dict):
    """Dictionary that defines the type of available models to be loaded.

    The order of classes must be asserted to the same defined by ClassType.

    """

    # Huggingface's Open AI GPT-2
    HF_GPT2 = ('HfGPT2', 'HfGPT2OnnxConfig', 'HfGPT2OnnxModel')

    # Huggingface's Transformer-XL
    HF_TRANSFO_XL = ('HfTransfoXL', 'HfTransfoXLOnnxConfig', 'HfTransfoXLOnnxModel')

    # NVIDIA's Memory Transfomer (Transformer-XL)
    MEM_TRANSFORMER = ('MemTransformerLM', 'MemTransformerLMOnnxConfig', 'MemTransformerLMOnnxModel')
