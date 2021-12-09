# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Lazy-loading utilities to import required classes
on demand.
"""

from enum import Enum
from importlib import import_module
from typing import Any, Dict, Optional

# Path to the `models` module
LIBRARY_PATH = 'archai.nlp.models'


class ClassType(Enum):
    """An enumerator that defines the type of available classes to be loaded.

    """

    # Types of classes
    CONFIG = 0
    ONNX_CONFIG = 1
    MODEL = 2
    ONNX_MODEL = 3


class ModelDict(Dict):
    """Dictionary that defines the type of available models to be loaded.

    The order of classes must be asserted to the same defined by ClassType.

    """

    # Huggingface's Open AI GPT-2
    HF_GPT2 = ('HfGPT2Config', 'HfGPT2OnnxConfig',
               'HfGPT2', 'HfGPT2OnnxModel')

    # Huggingface's Transformer-XL
    HF_TRANSFO_XL = ('HfTransfoXLConfig', 'HfTransfoXLOnnxConfig',
                     'HfTransfoXL', 'HfTransfoXLOnnxModel')

    # NVIDIA's Memory Transfomer (Transformer-XL)
    MEM_TRANSFORMER = ('MemTransformerConfig', 'MemTransformerLMOnnxConfig',
                       'MemTransformerLM', 'MemTransformerLMOnnxModel')


def load(model_type: str,
         cls_type: Optional[str] = 'config',
         **kwargs) -> Any:
    """Performs the lazy loading of a pre-defined model and its
        corresponding class.

    Args:
        model_type: Type of model to be loaded.
        cls_type: Type of class to be loaded.

    Returns:
        (Any): An instance of the loaded class.

    """

    # Transforms the input class type into an enumerator value
    cls_string = getattr(ClassType, cls_type.upper())

    # Finds the corresponding module based on the class
    if cls_string in [ClassType.CONFIG, ClassType.ONNX_CONFIG]:
        cls_module = import_module(f'.{model_type}.configs_{model_type}', LIBRARY_PATH)
    elif cls_string in [ClassType.MODEL]:
        cls_module = import_module(f'.{model_type}.model_{model_type}', LIBRARY_PATH)
    elif cls_string in [ClassType.ONNX_MODEL]:
        cls_module = import_module(f'{LIBRARY_PATH}.{model_type}.onnx_{model_type}')

    # Gathers the name of the class to be loaded
    cls_name = getattr(ModelDict, model_type.upper())

    # Attempts to load the class
    try:
        cls_instance = getattr(cls_module, cls_name[cls_string.value])
    except:
        raise ModuleNotFoundError

    # Initializes the class
    instance = cls_instance(**kwargs)

    return instance
