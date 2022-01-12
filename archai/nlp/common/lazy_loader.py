# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Lazy-loading utilities to import required classes
on demand.
"""

from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import torch

from archai.nlp.common.model_dict import ClassType, ModelDict

# Lazy loading constants
LIBRARY_PATH = 'archai.nlp.models'


def load_from_args(model_type: str,
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
    elif cls_string in [ClassType.CONFIG]:
        cls_module = import_module(f'.{model_type}.config_{model_type}', LIBRARY_PATH)
    elif cls_string in [ClassType.ONNX_MODEL, ClassType.ONNX_CONFIG]:
        cls_module = import_module(f'.{model_type}.onnx_{model_type}', LIBRARY_PATH)
    else:
        raise NotImplementedError

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


def load_model_from_checkpoint(model_type: str,
                               checkpoint_path: str,
                               replace_config: Optional[Dict[str, Any]] = None,
                               on_cpu: Optional[bool] = False,
                               for_export: Optional[bool] = False
                               ) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Performs the lazy loading of a pre-defined model and its configuration.

    Args:
        model_type: Type of model to be loaded.
        checkpoint_path: Path of the checkpoint to be loaded.
        replace_config: Dictionary with keys that should replace the model's configuration.
        on_cpu: Whether model should be loaded on CPU or not.
        for_export: If model should support export or not.

    Returns:
        (Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]): Model, configuration and checkpoint loaded from a checkpoint path.

    """

    # Gathers the proper device
    device = f'cuda:{torch.cuda.current_device()}' if not on_cpu and torch.cuda.is_available() else torch.device('cpu')

    # Finds the corresponding module based on the class
    model_cls_module = import_module(f'.{model_type}.model_{model_type}', LIBRARY_PATH)

    # Gathers the name of the class to be loaded
    cls_string = getattr(ClassType, 'model'.upper())
    cls_name = getattr(ModelDict, model_type.upper())

    # Attempts to load the class
    try:
        model_cls_instance = getattr(model_cls_module, cls_name[cls_string.value])
    except:
        raise ModuleNotFoundError

    # Loads the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    # Replaces keys that were provided in the `replace_config` dictionary
    if replace_config is not None:
        for k, v in replace_config.items():
            model_config[k] = v

    # Checks whether model is supposed to be exported
    if for_export:
        model_config['use_cache'] = True

    # Loads the model
    model = model_cls_instance(**model_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    return model, model_config, checkpoint
