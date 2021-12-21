# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Lazy-loading utilities to import required classes
on demand.
"""

from enum import Enum
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import torch

# Path to the `models` module
LIBRARY_PATH = 'archai.nlp.models'


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


def load(model_type: str,
         *args,
         cls_type: Optional[str] = 'model',
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
    if cls_string in [ClassType.MODEL]:
        cls_module = import_module(f'.{model_type}.model_{model_type}', LIBRARY_PATH)
    elif cls_string in [ClassType.ONNX_MODEL, ClassType.ONNX_CONFIG]:
        cls_module = import_module(f'{LIBRARY_PATH}.{model_type}.onnx_{model_type}')

    # Gathers the name of the class to be loaded
    cls_name = getattr(ModelDict, model_type.upper())

    # Attempts to load the class
    try:
        cls_instance = getattr(cls_module, cls_name[cls_string.value])
    except:
        raise ModuleNotFoundError

    # Initializes the class
    instance = cls_instance(*args, **kwargs)

    return instance


def load_from_checkpoint(model_type: str,
                         checkpoint_path: str,
                         on_cpu: Optional[bool] = False,
                         for_export: Optional[bool] = False) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Performs the lazy loading of a pre-defined model and its configuration.

    Args:
        model_type: Type of model to be loaded.
        checkpoint_path: Path of the checkpoint to be loaded.
        on_cpu: Whether model should be loaded on CPU or not.
        for_export: If model should support export or not.

    Returns:
        (Tuple[torch.nn.Module, Dict[str, Any]]): Model and its configuration loaded from a checkpoint.

    """

    # Gathers the proper device
    device = f'cuda:{torch.cuda.current_device()}' if not on_cpu and torch.cuda.is_available() else torch.device('cpu')

    # Finds the corresponding module based on the class
    model_cls_module = import_module(f'.{model_type}.model_{model_type}', LIBRARY_PATH)

    # Gathers the name of the class to be loaded
    cls_name = getattr(ModelDict, model_type.upper())

    # Attempts to load the class
    try:
        model_cls_instance = getattr(model_cls_module, cls_name['model'])
    except:
        raise ModuleNotFoundError

    # Loads the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    # Checks whether model is supposed to be exported
    if for_export:
        model_config['use_cache'] = True

    # Loads the model
    model = model_cls_instance(model_config)
    model.load_state_dict(checkpoint['model_state'])

    return model, model_config
