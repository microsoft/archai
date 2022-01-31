# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Loading utilities to import models and their configurations.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from archai.nlp.models.model_dict import MODELS, MODELS_CONFIGS, ONNX_MODELS, ONNX_MODELS_CONFIGS


def load_model_from_config(model_type: str, model_config: Dict[str, Any]) -> torch.nn.Module:
    """Performs the loading of a pre-defined model from configuration.

    Args:
        model_type: Type of model to be loaded.
        model_config: Configuration of model to be loaded.

    Returns:
        (torch.nn.Module): An instance of the loaded model.

    """

    return MODELS[model_type](**model_config)


def load_model_from_checkpoint(model_type: str,
                               checkpoint_path: str,
                               replace_model_config: Optional[Dict[str, Any]] = None,
                               on_cpu: Optional[bool] = False,
                               for_export: Optional[bool] = False) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Performs the loading of a pre-defined model from checkpoint file.

    Args:
        model_type: Type of model to be loaded.
        checkpoint_path: Path of the checkpoint to be loaded.
        replace_model_config: Dictionary with keys that should replace the model's configuration.
        on_cpu: Whether model should be loaded on CPU or not.
        for_export: If model should support export or not.

    Returns:
        (Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]): Model, configuration and checkpoint loaded from a checkpoint path.

    """

    # Gathers the proper device
    device = f'cuda:{torch.cuda.current_device()}' if not on_cpu and torch.cuda.is_available() else torch.device('cpu')

    # Loads the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    # Replaces keys that were provided in the `replace_config` dictionary
    if replace_model_config is not None:
        for k, v in replace_model_config.items():
            model_config[k] = v

    # Checks whether model is supposed to be exported
    if for_export:
        model_config['use_cache'] = True

    # Loads the model
    model = MODELS[model_type](**model_config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    return model, model_config, checkpoint


def load_model_config(model_type: str) -> Any:
    """
    """

    return MODELS_CONFIGS[model_type]()


def load_onnx_model(model_type: str) -> Any:
    """
    """

    return ONNX_MODELS[model_type]()


def load_onnx_model_config(model_type: str) -> Any:
    """
    """

    return ONNX_MODELS_CONFIGS[model_type]()
