# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pathlib
from typing import Any, Dict, Mapping

from omegaconf import OmegaConf
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorForLanguageModeling,
)


def _convert_nested_attrs_to_dict(attrs: Dict[str, Any]) -> Dict[str, Any]:
    def _attr_to_dict(key: str, value: Any) -> Dict[str, Any]:
        if len(key) == 1:
            return {key[0]: value}
        return _attr_to_dict(key[:-1], {key[-1]: value})

    def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = _deep_update(d.get(k, {}), v)
            else:
                d[k] = v

        return d

    converted_attrs = {}
    for key, value in attrs.items():
        _deep_update(converted_attrs, _attr_to_dict(key.split("."), value))

    return converted_attrs


def load_config(*configs) -> Dict[str, Any]:
    loaded_configs = []
    for cfg in configs:
        # If the config is a dictionary, we use OmegaConf.create()
        if isinstance(cfg, dict):
            loaded_configs.append(OmegaConf.create(cfg))
        
        # If the config is a list, we assume it is a dotlist
        if isinstance(cfg, list):
            loaded_configs.append(OmegaConf.from_dotlist(cfg))

        # If the config is a string, we parse it to find its extension
        if isinstance(cfg, str):
            file_extension = pathlib.Path(cfg).suffix

            # If the extension is .yaml, we use OmegaConf.load()
            if file_extension == ".yaml":
                loaded_configs.append(OmegaConf.load(cfg))

        # If the config is a argparse.Namespace, we convert it to a dictionary
        # and use OmegaConf.create()
        if isinstance(cfg, argparse.Namespace):
            cfg = _convert_nested_attrs_to_dict(vars(cfg))
            loaded_configs.append(OmegaConf.create(cfg))

    return OmegaConf.merge(*loaded_configs)


def load_collator(collator_name: str, **kwargs) -> DataCollator:
    AVAILABLE_DATA_COLLATORS = {"language_modelling": DataCollatorForLanguageModeling}

    collator_name = collator_name.replace("-", "_")
    available_collators = list(AVAILABLE_DATA_COLLATORS.keys())
    assert collator_name in available_collators, f"`collator_name` should be in {available_collators}."

    return AVAILABLE_DATA_COLLATORS[collator_name](**kwargs)
