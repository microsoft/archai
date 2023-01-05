# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Union

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from transformers.data.data_collator import (
    DataCollator,
    DataCollatorForLanguageModeling,
)


def from_yaml_file(yaml_file: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(yaml_file, "r") as f:
        output_dict = yaml.load(f, Loader=Loader)

    if output_dict is None:
        return {}

    return output_dict


def load_collator(collator_name: str, **kwargs) -> DataCollator:
    AVAILABLE_DATA_COLLATORS = {"language_modelling": DataCollatorForLanguageModeling}

    collator_name = collator_name.replace("-", "_")
    available_collators = list(AVAILABLE_DATA_COLLATORS.keys())
    assert collator_name in available_collators, f"`collator_name` should be in {available_collators}."

    return AVAILABLE_DATA_COLLATORS[collator_name](**kwargs)
