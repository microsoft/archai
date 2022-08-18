# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""External data format parsers.
"""

import json
import os
from json.decoder import JSONDecodeError
from typing import Any, Dict, Optional, Union

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def from_json_file(json_file: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Loads a JSON file into a dictionary.

    Args:
        json_file: Path to the JSON to be loaded.

    Returns:
        (Dict[str, Any]): Dictionary mapping the object from the JSON file.

    """

    with open(json_file, "r") as f:
        try:
            output_dict = json.load(f)
        except JSONDecodeError:
            output_dict = None

    if output_dict is None:
        return {}

    return output_dict


def to_json_file(input_dict: Dict[str, Any], json_file: Union[str, os.PathLike]) -> None:
    """Outputs a dictionary into a JSON file.

    Args:
        input_dict: Dictionary to be mapped to file.
        json_file: Path to the jsJSONon to be saved.

    """

    with open(json_file, "w") as f:
        json.dump(input_dict, f)


def from_yaml_file(yaml_file: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Loads a YAML file into a dictionary.

    Args:
        yaml_file: Path to the YAML to be loaded.

    Returns:
        (Dict[str, Any]): Dictionary mapping the object from the YAML file.

    """

    with open(yaml_file, "r") as f:
        output_dict = yaml.load(f, Loader=Loader)

    if output_dict is None:
        return {}

    return output_dict


def to_yaml_file(
    input_dict: Dict[str, Any],
    yaml_file: Union[str, os.PathLike],
    default_flow_style: Optional[bool] = False,
) -> None:
    """Outputs a dictionary into a YAML file.

    Args:
        input_dict: Dictionary to be mapped to file.
        yaml_file: Path to the YAML to be saved.
        default_flow_style: Whether to output YAML using flow style.

    """

    with open(yaml_file, "w") as f:
        yaml.dump(input_dict, f, default_flow_style=default_flow_style)
