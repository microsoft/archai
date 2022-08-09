# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable automatic Transformer-based models.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import torch
from transformers.models.auto.configuration_auto import AutoConfig

# from archai_nlp.utils import logging_utils

# logger = logging_utils.get_logger(__name__)

# Available models mapping
MODELS = {
    "opt": {"clm": ("OPTConfig", "OPTForCLM")},
    "opt_flex": {"clm": ("OPTFlexConfig", "OPTFlexForCLM")},
}


def get_layers_from_module(
    module: torch.nn.Module, layer_type: Optional[str] = None
) -> List[torch.nn.Module]:
    """Gets a list of children layers from an input module.

    Args:
        module: Module to retrieve layers.
        layer_type: Type of layers to be retrieved.

    Returns:
        (List[torch.nn.Module]): List of layers from input module.

    """

    sub_module = list(module.children())
    layers = []

    if layer_type is not None:
        for lt in layer_type:
            if module.__class__.__name__ == lt:
                return module
    else:
        if len(sub_module) == 0 and len(list(module.parameters())) > 0:
            return module

    for m in sub_module:
        try:
            layers.extend(get_layers_from_module(m, layer_type))
        except TypeError:
            layers.append(get_layers_from_module(m, layer_type))

    return layers


def get_params_from_module(module: torch.nn.Module, layer_type: str) -> int:
    """Gets the number of parameters from an input module.

    Args:
        module: Module to retrieve parameters.
        layer_type: Type of layer.

    Returns:
        (int): Number of parameters from input module.

    """

    layers = get_layers_from_module(module, layer_type)
    n_params = {}

    for i, layer in enumerate(layers):
        layer_name = layer.__class__.__name__ + "_" + str(i)
        n_params[layer_name] = sum([p.nelement() for p in layer.parameters()])

    return sum(list(n_params.values()))


class ArchaiModel:
    """Provides an AutoModel-like utility to instantiate new models and load pre-trained ones."""

    @property
    def n_parameters(self) -> int:
        """Calculates the number of total parameters of the model.

        Returns:
            (int): Number of total parameters.

        """

        return sum(p.numel() for p in self.parameters())

    @property
    def n_embedding_parameters(self) -> int:
        """Calculates the number of embedding parameters of the model.

        Returns:
            (int): Number of embedding parameters.

        """

        return get_params_from_module(self, ["Embedding"])

    @property
    def n_non_embedding_parameters(self) -> int:
        """Calculates the number of non-embedding parameters of the model.

        Returns:
            (int): Number of non-embedding parameters.

        """

        return self.n_parameters - self.n_embedding_parameters

    @classmethod
    def from_config(cls: ArchaiModel, **kwargs) -> ArchaiModel:
        """Instantiates a new model from a configuration object.

        Returns:
            (ArchaiModel): A PyTorch model wrapped into a transformer-based class.

        """

        model_type = kwargs["model_type"].replace("-", "_")
        if model_type in MODELS.keys():
            models = MODELS[model_type]
        else:
            raise NotImplementedError(f"model: {model_type} has not been implemented yet.")

        model_task = kwargs["model_task"].replace("-", "_")
        if model_task in models.keys():
            model = models[model_task]
        else:
            raise NotImplementedError(
                f"model_task: {model_task} has not been implemented yet."
            )

        config_cls_name = model[0]
        model_cls_name = model[1]
        model_module = importlib.import_module("archai.nlp.models")
        config_cls = getattr(model_module, config_cls_name)
        model_cls = getattr(model_module, model_cls_name)

        # logger.info(f"Loading model from: {config_cls_name}")
        # logger.debug(kwargs)

        model = model_cls(config=config_cls(**kwargs))

        # logger.info("Model loaded.")

        return model

    @classmethod
    def from_pretrained(
        cls: ArchaiModel,
        pre_trained_model_path: str,
        *args,
        replace_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ArchaiModel:
        """Instantiates a new model from a pre-trained checkpoint.

        Args:
            pre_trained_model_path: Path to the pre-trained checkpoint.
            replace_config: Configuration that should be replaced, such as new task for fine-tune.

        Returns:
            (ArchaiModel): A PyTorch model wrapped into a transformer-based class.

        """

        # Loads an AutoConfig object from the pre-trained model
        # and replaces any potential new configuration passed in as keywords
        config = AutoConfig.from_pretrained(pre_trained_model_path, **kwargs)

        replace_config = replace_config or {}
        for k, v in replace_config.items():
            setattr(config, k, v)

        model_type = config.model_type.replace("-", "_")
        if model_type in MODELS.keys():
            models = MODELS[model_type]
        else:
            raise NotImplementedError(f"model: {model_type} has not been implemented yet.")

        model_task = (
            config.model_task.replace("-", "_")
            if getattr(config, "model_task", None)
            else "clm"
        )
        if model_task in models.keys():
            model = models[model_task]
        else:
            raise NotImplementedError(
                f"model_task: {model_task} has not been implemented yet."
            )

        model_cls_name = model[1]
        model_module = importlib.import_module("archai.nlp.models")
        model_cls = getattr(model_module, model_cls_name)

        # logger.info(f"Loading model from: {pre_trained_model_path}")

        model = model_cls.from_pretrained(
            pre_trained_model_path, *args, config=config, **kwargs
        )

        # logger.info("Model loaded.")

        return model
