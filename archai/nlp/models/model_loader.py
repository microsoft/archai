# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Functions that allows easy-loading of models and their configurations.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from onnxruntime.transformers.onnx_model import OnnxModel

from archai.nlp.models.config_base import Config, OnnxConfig, SearchConfig
from archai.nlp.models.model_base import ArchaiModel
from archai.nlp.models.model_dict import (MODELS, MODELS_CONFIGS,
                                          MODELS_SEARCH_CONFIGS, MODELS_PARAMS_FORMULAE,
                                          ONNX_MODELS, ONNX_MODELS_CONFIGS)


def load_model_formula(model_type: str) -> Callable:
    """Loads an available analytical parameters formula.

    Args:
        model_type: Type of the model.

    Returns:
        (Callable): Function that analytically calculates parameters.
        
    """

    if model_type not in MODELS_PARAMS_FORMULAE.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return MODELS_PARAMS_FORMULAE[model_type]


def load_model_from_config(model_type: str, model_config: Dict[str, Any]) -> ArchaiModel:
    """Loads an available model from a configuration dictionary.

    Args:
        model_type: Type of the model.
        model_config: Configuration of the model that will be created.

    Returns:
        (ArchaiModel): An instance of the created model.

    """

    if model_type not in MODELS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return MODELS[model_type](**model_config)


def load_model_from_checkpoint(model_type: str,
                               checkpoint_path: str,
                               replace_model_config: Optional[Dict[str, Any]] = None,
                               on_cpu: Optional[bool] = False,
                               for_export: Optional[bool] = False
                               ) -> Tuple[ArchaiModel, Dict[str, Any], Dict[str, Any]]:
    """Loads an available model from a pre-trained checkpoint.

    Args:
        model_type: Type of the model.
        checkpoint_path: Path to the pre-trained checkpoint.
        replace_model_config: Model's configuration replacement dictionary.
        on_cpu: Whether model should be loaded to CPU.
        for_export: Whether model should be ready for ONNX exporting.

    Returns:
         (Tuple[ArchaiModel, Dict[str, Any], Dict[str, Any]]): Model, configuration
            and checkpoint dictionaries.

    """

    device = f'cuda:{torch.cuda.current_device()}' if not on_cpu and torch.cuda.is_available() else torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    if replace_model_config is not None:
        for k, v in replace_model_config.items():
            model_config[k] = v

    if for_export:
        model_config['use_cache'] = True

    model = load_model_from_config(model_type, model_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    return model, model_config, checkpoint


def load_config(model_type: str) -> Config:
    """Loads an available default configuration class.

    Args:
        model_type: Type of the model.
    
    Returns:
        (Config): Configuration.

    """

    if model_type not in MODELS_CONFIGS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return MODELS_CONFIGS[model_type]()


def load_search_config(model_type: str) -> SearchConfig:
    """Loads an available search configuration class.

    Args:
        model_type: Type of the model.
    
    Returns:
        (SearchConfig): Search configuration.

    """

    if model_type not in MODELS_SEARCH_CONFIGS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return MODELS_SEARCH_CONFIGS[model_type]()


def load_onnx_model(model_type: str, *model_args) -> OnnxModel:
    """Loads an available ONNX-based model (used during export optimization).

    Args:
        model_type: Type of the model.

    Returns:
        (OnnxModel): ONNX-based optimization model.

    """

    if model_type not in ONNX_MODELS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return ONNX_MODELS[model_type](*model_args)


def load_onnx_config(model_type: str, model_config: Dict[str, Any]) -> OnnxConfig:
    """Loads an available ONNX-based configuration (used during export).

    Args:
        model_type: Type of the model.
        model_config: Model's configuration used to supply missing attributes.

    Returns:
        (OnnxConfig): ONNX-based configuration.

    """

    if model_type not in ONNX_MODELS_CONFIGS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return ONNX_MODELS_CONFIGS[model_type](model_config)
