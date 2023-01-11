import json
import os
from json.decoder import JSONDecodeError
from typing import Any, Dict, Union
from itertools import chain

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def from_json_file(json_file: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(json_file, "r") as f:
        try:
            output_dict = json.load(f)
        except JSONDecodeError:
            output_dict = None

    if output_dict is None:
        return {}

    return output_dict


def from_yaml_file(yaml_file: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(yaml_file, "r") as f:
        output_dict = yaml.load(f, Loader=Loader)

    if output_dict is None:
        return {}

    return output_dict


def group_texts(examples, tokenizer, **kwargs):
        block_size = tokenizer.model_max_length
        
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
