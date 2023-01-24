# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
import copy
import os
from collections import UserDict
from distutils.util import strtobool
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional

import yaml

from archai.common.config_utils import resolve_all


def deep_update(d: MutableMapping, u: Mapping, mapping_fn: Callable[[], MutableMapping]) -> MutableMapping:
    """Recursively update a dictionary with another dictionary.

    Args:
        d: Dictionary to update.
        u: Dictionary to update from.
        mapping_fn: Dunction to create a new dictionary.

    Returns:
        Updated dictionary.

    """

    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, mapping_fn()), v, mapping_fn)
        else:
            d[k] = v

    return d


class Config(UserDict):
    """Configuration class that supports YAML-based configuration files and
    command line arguments.

    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        args: Optional[List[Any]] = None,
        parse_cl_args: Optional[bool] = False,
        cl_description: Optional[str] = None,
        resolve_redirect: Optional[bool] = True,
    ) -> None:
        """Initialize a configuration object.

        Args:
            file_path: Path to YAML-based configuration file.
            args: List of extra arguments, e.g., ["--arg", "value", "--arg2", "value2"].
            parse_cl_args: Whether to parse command line arguments.
            cl_description: Description to use for command line arguments.
            resolve_redirect: Whether to resolve the YAML-based redirects.

        """

        super().__init__()

        args = args or []
        extra_args = []

        if parse_cl_args:
            parser = argparse.ArgumentParser(description=cl_description)
            parser.add_argument(
                "-c" "--config",
                type=str,
                default=None,
                help="YAML-based configuration file-paths (separated by `,` if multiple).",
            )

            args, extra_args = parser.parse_known_args()
            file_path = args.config or file_path

        if file_path:
            for fp in file_path.strip().split(","):
                self._load(fp.strip())

        # Create a copy and resolve it, which can be used to search for overrides
        # that would not have existed before resolution
        r_config = copy.deepcopy(self)
        if resolve_redirect:
            resolve_all(r_config)

        # Update with additional arguments
        self._update_from_args(args, r_config)  # Merge from supplied args
        self._update_from_args(extra_args, r_config)  # Merge from command line args

        if resolve_redirect:
            resolve_all(self)

        self.file_path = file_path

    def _load(self, file_path: str) -> None:
        """Load a YAML-based configuration file.

        Args:
            file_path: Path to YAML-based configuration file.

        """

        if file_path:
            file_path = os.path.abspath(os.path.expanduser(os.path.expandvars(file_path)))
            with open(file_path, "r") as f:
                config_yaml = yaml.load(f, Loader=yaml.Loader)

            self._update_from_include(config_yaml, file_path)
            deep_update(self, config_yaml, lambda: Config(resolve_redirect=False))

    def _update_from_include(self, config_yaml: Dict[str, Any], file_path: str) -> None:
        """Update the configuration from the __include__ directive.

        Args:
            config_yaml: YAML-based configuration.
            file_path: Path to YAML-based configuration file.

        """

        if "__include__" in config_yaml:
            includes = config_yaml["__include__"]
            if isinstance(includes, str):
                includes = [includes]

            for include in includes:
                include_file_path = os.path.join(os.path.dirname(file_path), include)
                self._load_from_file(include_file_path)

    def _update_from_args(self, args: List[Any], resolved_section: Config) -> None:
        """Update the configuration from extra/command line arguments.

        Args:
            args: List of extra arguments, e.g., ["--arg", "value", "--arg2", "value2"].
            resolved_section: Resolved configuration.

        """

        i = 0
        while i < len(args) - 1:
            arg = args[i]
            if arg.startswith("--"):
                path = arg[2:].split(".")
                i += self._update_section(path, args[i + 1], resolved_section)
            else:
                i += 1

    def _update_section(self, path: List[str], value: Any, resolved_section: Config) -> int:
        """Update the section of a configuration object.

        Args:
            path: Path to the configuration section.
            value: Value to set.
            resolved_section: Resolved configuration.

        Returns:
            `2` if arguments have been consumed and `1` if path has not been found.

        """

        for p in range(len(path) - 1):
            sub_path = path[p]
            if sub_path in resolved_section:
                resolved_section = resolved_section[sub_path]

            if sub_path not in self:
                self[sub_path] = Config(resolve_redirect=False)
            self = self[sub_path]

        key = path[-1]  # Final leaf node value
        if key in resolved_section:
            original_val, original_type = None, None
            try:
                original_val = resolved_section[key]
                original_type = type(original_val)

                if original_type == bool:  # bool('False') is True :(

                    def original_type(x):
                        return strtobool(x) == 1

                self[key] = original_type(value)

            except Exception:
                raise KeyError(
                    f"Key: {key} could not been set to value {value}."
                    f"Original value was {original_val} of type {original_type}."
                )
        else:
            self[key] = value

        return 2  # Path was found or created, increment arg pointer by 2 as we use up val

    def to_dict(self) -> Dict[str, Any]:
        """Convert the `Config` object to a dictionary.

        Returns:
            `Config` represented as a dictionary.

        """

        return deep_update({}, self, lambda: dict())

    def get(self, key: str, default_value: Optional[Any] = None) -> Any:
        """Get a value from the `Config` object.

        Args:
            key: Key to search for.
            default_value: Default value to return if `key` is not found.

        Returns:
            Value corresponding to the key.

        """

        return super().get(key, default_value)
