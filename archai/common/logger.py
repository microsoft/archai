# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import itertools
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import yaml

from archai.common.logger_utils import get_logger


class Logger:
    """"""

    def __init__(
        self, source: Optional[str] = None, file_path: Optional[str] = None, delay: Optional[float] = 30.0
    ) -> None:
        """"""

        self.logger = get_logger(source or __name__)

        self.file_path = file_path or "archai.log.yaml"
        self.delay = delay

        self.call_count = 0
        self.timestamp = time.time()

        self.paths = [[""]]
        self.stack = [OrderedDict()]

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.popd()

    def __contains__(self, key: str) -> bool:
        return key in self.current_node

    def __len__(self) -> int:
        return len(self.current_node)

    @property
    def root_node(self) -> OrderedDict:
        """"""

        return self.stack[0]

    @property
    def current_node(self) -> OrderedDict:
        """"""

        last_obj = None

        for i, (path, obj) in enumerate(zip(self.paths, self.stack)):
            if obj is None:
                obj = last_obj

                for key in path:
                    if key not in obj:
                        obj[key] = OrderedDict()
                    if not isinstance(obj[key], OrderedDict):
                        raise RuntimeError(f"`{key}` is being used to store a scalar value.")
                    obj = obj[key]

                self.stack[i] = obj

            last_obj = obj

        return self.stack[-1]

    @property
    def current_path(self) -> str:
        return "/".join(itertools.chain.from_iterable(self.paths[1:]))

    def save(self) -> None:
        with open(self.file_path, "w") as f:
            yaml.dump(self.root_node, f)

    def load(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            obj = yaml.load(f, Loader=yaml.Loader)
            self.stack = [obj]

    def update_key(
        self,
        key: Any,
        value: Any,
        node: Optional[OrderedDict] = None,
        path: Optional[List[str]] = None,
        override_key: Optional[bool] = False,
    ) -> None:
        if not override_key and key in self.current_node:
            raise KeyError(f"`{key}` is already being used. Cannot use it again, unless popd() is called.")

        current_node = node or self.current_node
        current_path = path or []

        for p in current_path:
            if p not in current_node:
                current_node[p] = OrderedDict()
            current_node = current_node[p]
        current_node[str(key)] = value

    def update(self, obj: Dict[str, Any], override_key: Optional[bool] = False) -> None:
        """"""

        for k, v in obj.items():
            self.update_key(k, v, override_key=override_key)

    def log(
        self, obj: Union[Dict[str, Any], str], level: Optional[int] = None, override_key: Optional[bool] = False
    ) -> None:
        """"""

        self.call_count += 1

        if isinstance(obj, dict):
            self.update(obj, override_key=override_key)
            message = ", ".join(f"{k}={v}" for k, v in obj.items())
        else:
            message = obj
            path = {
                logging.INFO: ["messages"],
                logging.DEBUG: ["debugs"],
                logging.WARNING: ["warnings"],
                logging.ERROR: ["errors"],
            }
            self.update_key(self.call_count, message, node=self.root_node, path=path[level], override_key=override_key)

        self.logger.log(msg=self.current_path + " " + message, level=level)

        if time.time() - self.timestamp > self.delay:
            self.save()
            self.timestamp = time.time()

    def info(self, obj: Dict[str, Any], override_key: Optional[bool] = False) -> None:
        """"""

        self.log(obj, level=logging.INFO, override_key=override_key)

    def debug(self, obj: Dict[str, Any], override_key: Optional[bool] = False) -> None:
        """"""

        self.log(obj, level=logging.DEBUG, override_key=override_key)

    def warn(self, obj: Dict[str, Any], override_key: Optional[bool] = False) -> None:
        """"""

        self.log(obj, level=logging.WARNING, override_key=override_key)

    def error(self, obj: Dict[str, Any], override_key: Optional[bool] = False) -> None:
        """"""

        self.log(obj, level=logging.ERROR, override_key=override_key)

    def pushd(self, *keys: Any) -> Logger:
        """"""

        self.paths.append([str(k) for k in keys])
        self.stack.append(None)  # Delays creation of node until it is needed

        return self  # Allows to call __enter__

    def popd(self) -> None:
        """"""

        if len(self.stack) == 1:
            self.warn("Invalid call due to no child logger.")
            return

        self.stack.pop()
        self.paths.pop()
