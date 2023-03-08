# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def build_arch_config(config_dict: Dict[str, Any]) -> ArchConfig:
    """Build an `ArchConfig` object from a sampled config dictionary.

    Args:
        config_dict: Config dictionary

    Returns:
        `ArchConfig` object.

    """

    ARCH_CONFIGS = {"default": ArchConfig, "config_list": ArchConfigList}

    config_type = config_dict.get("_config_type", "default")
    return ARCH_CONFIGS[config_type](config_dict)


class ArchConfig:
    """Store architecture configs."""

    def __init__(self, config_dict: Dict[str, Union[dict, float, int, str]]) -> None:
        """Initialize the class.

        Args:
            config_dict: Configuration dictionary.

        """

        # Set that stores all parameters used to build the model instance
        self._used_params = set()

        # Original config dictionary
        self._config_dict = deepcopy(config_dict)

        # ArchConfig nodes
        self.nodes = OrderedDict()

        for param_name, param in self._config_dict.items():
            if isinstance(param, dict):
                self.nodes[param_name] = build_arch_config(param)
            else:
                self.nodes[param_name] = param

    def __repr__(self) -> str:
        class ArchConfigJsonEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, ArchConfig):
                    return o.to_dict(remove_metadata_info=True)

                return super().default(o)

        cls_name = self.__class__.__name__
        return f"{cls_name}({json.dumps(self, cls=ArchConfigJsonEncoder, indent=4)})"

    def get_used_params(self) -> Dict[str, Union[Dict, bool]]:
        """Get the parameter usage tree.

        Terminal nodes with value `True` represent architecture parameters that were used
        by calling `ArchConfig.pick(param_name)`.

        Returns:
            Used parameters.
        """

        used_params = OrderedDict()

        for param_name, param in self.nodes.items():
            used_params[param_name] = param_name in self._used_params

            if isinstance(param, ArchConfig):
                used_params[param_name] = param.get_used_params()

        return used_params

    def pick(self, param_name: str, default: Optional[Any] = None, record_usage: Optional[bool] = True) -> Any:
        """Pick an architecture parameter, possibly recording its usage.

        Args:
            param_name: Architecture parameter name
            default: Default value to return if parameter is not found. If `None`, an
                exception is raised.
            record_usage: If this parameter should be recorded as 'used' in
                `ArchConfig._used_params`.

        Returns:
            Parameter value.

        """
        if param_name in self.nodes:
            param_value = self.nodes[param_name]
        else:
            if default is None:
                raise ValueError(
                    f"Architecture parameter {param_name} not found in config and "
                    f"no default value provided. Available parameters are: {self.nodes.keys()}"
                )
            param_value = default

        if record_usage:
            self._used_params.add(param_name)

        return param_value

    def to_dict(self, remove_metadata_info: Optional[bool] = False) -> OrderedDict:
        """Convert `ArchConfig` object to an ordered dictionary.

        Args:
            remove_metadata_info: If keys used to store extra metadata should be removed.

        Returns:
            Ordered dictionary.

        """

        return OrderedDict(
            (k, v.to_dict(remove_metadata_info)) if isinstance(v, ArchConfig) else (k, v)
            for k, v in self.nodes.items()
            if not remove_metadata_info or not k.startswith("_")
        )

    def to_file(self, path: str) -> None:
        """Save `ArchConfig` object to a file.

        Args:
            path: Path to save the file to.

        """

        path = Path(path)
        path = path.parent / f"{path.name}.json" if path.suffix == "" else path

        d = self.to_dict()

        if path.suffix == ".yaml":
            yaml.dump(d, open(path, "w", encoding="utf-8"), default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            json.dump(d, open(path, "w", encoding="utf-8"), indent=4)
        else:
            raise ValueError(f"Unsupported file extension {path.suffix}")

    @classmethod
    def from_file(cls, path: str) -> ArchConfig:
        """Load `ArchConfig` object from a file.

        Args:
            path: Path to load the file from.

        Returns:
            `ArchConfig` object.

        """

        path = Path(path)
        path = path.parent / f"{path.name}.json" if path.suffix == "" else path

        if path.suffix == ".yaml":
            d = yaml.load(open(path, "r", encoding="utf-8"), Loader=yaml.Loader)
        elif path.suffix == ".json":
            d = json.load(open(path, "r", encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported file extension {path.suffix}")
        
        return build_arch_config(d)


class ArchConfigList(ArchConfig):
    """Store a list of architecture configs."""

    def __init__(self, config: OrderedDict):
        """Initialize the class.

        Args:
            config: Configuration dictionary.

        """

        super().__init__(config)

        assert "_configs" in config
        assert "_repeat_times" in config

        self.max_size = config["_repeat_times"]

    def __len__(self) -> int:
        self._used_params.add("_repeat_times")
        return self.max_size

    def __getitem__(self, idx: int) -> ArchConfig:
        if 0 <= idx < len(self):
            self._used_params.add("_repeat_times")
            return self.nodes["_configs"].pick(str(idx))
        raise IndexError

    def __iter__(self):
        yield from [self[i] for i in range(len(self))]

    def pick(self, param_name: str, record_usage: Optional[bool] = True) -> None:
        raise ValueError(
            "Attempted to use .pick in an ArchConfigList instance. "
            "Select a config first using indexing (e.g `config_list[i]`)."
        )

    def to_dict(self, remove_metadata_info: Optional[bool] = False) -> OrderedDict:
        if remove_metadata_info:
            return [
                self.nodes["_configs"].pick(str(i), record_usage=False).to_dict(remove_metadata_info)
                for i in range(self.max_size)
            ][:self.max_size]

        return super().to_dict(remove_metadata_info)
