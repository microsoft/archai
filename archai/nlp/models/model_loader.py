# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Loading utilities to import models and their configurations.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from archai.nlp.models.model_dict import (MODELS, MODELS_CONFIGS, MODELS_PARAMS_FORMULAE,
                                          ONNX_MODELS, ONNX_MODELS_CONFIGS)


def load_model_formula(model_type: str) -> Callable:
    if model_type not in MODELS_CONFIGS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return MODELS_PARAMS_FORMULAE[model_type]


def load_model_from_checkpoint(model_type: str,
                               checkpoint_path: str,
                               replace_model_config: Optional[Dict[str, Any]] = None,
                               on_cpu: Optional[bool] = False,
                               for_export: Optional[bool] = False) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    device = f'cuda:{torch.cuda.current_device()}' if not on_cpu and torch.cuda.is_available() else torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    if replace_model_config is not None:
        for k, v in replace_model_config.items():
            model_config[k] = v

    if for_export:
        model_config['use_cache'] = True

    model = MODELS[model_type](**model_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    return model, model_config, checkpoint


def load_model_from_config(model_type: str, model_config: Dict[str, Any]) -> torch.nn.Module:
    if model_type not in MODELS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return MODELS[model_type](**model_config)


def load_config(model_type: str, config_type: Optional[str] = 'default') -> Any:
    if model_type not in MODELS_CONFIGS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')
        
    if config_type not in ['default', 'search']:
        raise Exception(f'config_type: {config_type} not supported yet.')

    return getattr(MODELS_CONFIGS[model_type](), config_type)


def load_onnx_model(model_type: str, *model_args) -> Any:
    if model_type not in ONNX_MODELS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return ONNX_MODELS[model_type](*model_args)


def load_onnx_config(model_type: str, model_config: Dict[str, Any]) -> Any:
    if model_type not in ONNX_MODELS_CONFIGS.keys():
        raise Exception(f'model_type: {model_type} not supported yet.')

    return ONNX_MODELS_CONFIGS[model_type](model_config)
