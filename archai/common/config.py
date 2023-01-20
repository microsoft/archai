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
    """"""

    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, mapping_fn()), v, mapping_fn)
        else:
            d[k] = v

    return d


class Config(UserDict):
    """"""

    def __init__(
        self,
        file_path: Optional[str] = None,
        args: Optional[List[Any]] = None,
        force_cl_args: Optional[bool] = False,
        cl_description: Optional[str] = None,
        resolve_redirect: Optional[bool] = True,
    ) -> None:
        """"""

        super().__init__()

        args = args or []
        extra_args = []

        if force_cl_args:
            parser = argparse.ArgumentParser(description=cl_description)
            parser.add_argument(
                "-c" "--config",
                type=str,
                default=None,
                help="YAML-based configuration file-paths (separated by `;` if multiple).",
            )

            args, extra_args = parser.parse_known_args()
            file_path = args.config or file_path

        if file_path:
            for fp in file_path.strip().split(";"):
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
        """"""

        if file_path:
            file_path = os.path.abspath(os.path.expanduser(os.path.expandvars(file_path)))
            with open(file_path, "r") as f:
                config_yaml = yaml.load(f, Loader=yaml.Loader)

            self._update_from_include(config_yaml, file_path)
            deep_update(self, config_yaml, lambda: Config(resolve_redirect=False))

    def _update_from_include(self, config_yaml: Dict[str, Any], file_path: str) -> None:
        """"""

        if "__include__" in config_yaml:
            includes = config_yaml["__include__"]
            if isinstance(includes, str):
                includes = [includes]

            for include in includes:
                include_file_path = os.path.join(os.path.dirname(file_path), include)
                self._load_from_file(include_file_path)

    def _update_from_args(self, args: List[Any], resolved_section: Config) -> None:
        """"""

        i = 0
        while i < len(args) - 1:
            arg = args[i]
            if arg.startswith(("--")):
                path = arg[len("--") :].split(".")
                i += self._update_section(path, args[i + 1], resolved_section)
            else:
                i += 1

    def _update_section(self, path: List[str], value: Any, resolved_section: Config) -> int:
        """"""

        for p in range(len(path) - 1):
            sub_path = path[p]
            if sub_path in resolved_section:
                resolved_section = resolved_section[sub_path]
                if sub_path not in self:
                    self[sub_path] = Config(resolve_redirects=False)
                self = self[sub_path]
            else:
                return 1  # path not found, ignore this
        key = path[-1]  # final leaf node value

        if key in resolved_section:
            original_val, original_type = None, None
            try:
                original_val = resolved_section[key]
                original_type = type(original_val)
                if original_type == bool:  # bool('False') is True :(

                    def original_type(x):
                        return strtobool(x) == 1

                self[key] = original_type(value)
            except Exception as e:
                raise KeyError(
                    f'The yaml key or command line argument "{key}" is likely not named correctly or value is of wrong data type. Error was occured when setting it to value "{value}".'
                    f"Originally it is set to {original_val} which is of type {original_type}."
                    f"Original exception: {e}"
                )
            return 2  # path was found, increment arg pointer by 2 as we use up val
        else:
            return 1  # path not found, ignore this

    def to_dict(self) -> Dict[str, Any]:
        """"""

        return deep_update({}, self, lambda: dict())

    def get(self, key: str, default_value: Optional[Any] = None) -> Any:
        """"""

        return super().get(key, default_value)
