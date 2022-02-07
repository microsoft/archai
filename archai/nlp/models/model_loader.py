# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Loading utilities to import models and their configurations on demand.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from archai.nlp.models.model_dict import (MODELS, MODELS_CONFIGS, MODELS_PARAMS_FORMULAE,
                                          ONNX_MODELS, ONNX_MODELS_CONFIGS)


def load_model_formula(model_type: str) -> Callable:
    if model_type not in MODELS_CONFIGS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return MODELS_PARAMS_FORMULAE[model_type]

# Path to the `models` package
PACKAGE_PATH = 'archai.nlp.models'


def load_from_args(model_type: str, *args, cls_type: Optional[str] = 'model', **kwargs) -> Any:
    """Performs the loading of a pre-defined model and its
        corresponding class.

    Args:
        model_type: Type of model to be loaded.
        cls_type: Type of class to be loaded.

    Returns:
        (Any): An instance of the loaded class.

    """

    # Gathers the name and index of corresponding type of class
    cls_type = getattr(ModelClassType, cls_type.upper())
    cls_type_idx = cls_type.value

    # Gathers the available tuple to be loaded and its corresponding class name
    cls_tuple = getattr(ModelDict, model_type.upper())
    cls_name = cls_tuple[cls_type_idx]

    # Finds the corresponding module based on the class
    if cls_type in [ModelClassType.MODEL]:
        cls_module = import_module(f'.{model_type}.model_{model_type}', PACKAGE_PATH)
    elif cls_type in [ModelClassType.CONFIG]:
        cls_module = import_module(f'.{model_type}.config_{model_type}', PACKAGE_PATH)
    elif cls_type in [ModelClassType.ONNX_MODEL, ModelClassType.ONNX_CONFIG]:
        cls_module = import_module(f'.{model_type}.onnx_{model_type}', PACKAGE_PATH)
    else:
        raise ModuleNotFoundError

    # Attempts to load the class
    try:
        cls_instance = getattr(cls_module, cls_name)
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
    """Performs the loading of a pre-defined model and its configuration.

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

    # Gathers the name and index of corresponding type of class
    cls_type = ModelClassType.MODEL
    cls_type_idx = cls_type.value

    # Gathers the available tuple to be loaded and its corresponding class name
    cls_tuple = getattr(ModelDict, model_type.upper())
    cls_name = cls_tuple[cls_type_idx]

    # Attempts to load the class
    try:
        cls_module = import_module(f'.{model_type}.model_{model_type}', PACKAGE_PATH)
        cls_instance = getattr(cls_module, cls_name)
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
    model = cls_instance(**model_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    return model, model_config, checkpoint
