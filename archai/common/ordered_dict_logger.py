# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import itertools
import logging
import os
import pathlib
import time
from collections import OrderedDict
from types import TracebackType
from typing import Any, Dict, List, Optional, Union

import yaml

from archai.common.ordered_dict_logger_utils import get_logger


class OrderedDictLogger:
    """Log and save data in a hierarchical YAML structure.
    
    The purpose of the structured logging is to store logs as key value pair.
    However, when you have loop and sub routine calls, what you need is hierarchical
    dictionaries where the value for a key could be a dictionary. The idea is that you
    set one of the nodes in tree as current node and start logging your values. You can
    then use pushd to create and go to child node and popd to come back to parent.
    
    To implement this mechanism we use two main variables: _stack allows us to push each node
    on stack when pushd is called. The node is OrderedDictionary. As a convinience, we let
    specify child path in pushd in which case child hierarchy is created and current node
    will be set to the last node in specified path. When popd is called, we go back to
    original parent instead of parent of current node. To implement this we use _paths 
    variable which stores subpath when each pushd call was made.

    """

    def __init__(
        self, source: Optional[str] = None, file_path: Optional[str] = None, delay: Optional[float] = 60.0
    ) -> None:
        """Initialize the logger.

        Args:
            source: Source of the logger.
            file_path: File path of the log file.
            delay: Delay between log saves.

        """

        self.logger = get_logger(source or __name__)

        self.file_path = file_path
        self.delay = delay

        self.call_count = 0
        self.timestamp = time.time()

        self.paths = [[""]]
        self.stack = [OrderedDict()]

        if self.file_path:
            if os.path.exists(self.file_path):
                backup_file_path = pathlib.Path(self.file_path)
                backup_file_path.rename(backup_file_path.with_suffix(f".{str(int(time.time()))}.yaml"))

    def __enter__(self) -> OrderedDictLogger:
        return self

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        self.popd()

    def __contains__(self, key: str) -> bool:
        return key in self.current_node

    def __len__(self) -> int:
        return len(self.current_node)

    @property
    def root_node(self) -> OrderedDict:
        """Return the root node of the current stack."""

        return self.stack[0]

    @property
    def current_node(self) -> OrderedDict:
        """Return the current node of the current stack.

        Raises:
            RuntimeError: If a `key` stores a scalar value and is trying to store
                new information.

        """

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
        """Return the current path of the current stack."""

        return "/".join(itertools.chain.from_iterable(self.paths[1:]))

    def save(self) -> None:
        """Save the current log data to an output file.

        This method only saves to a file if a valid `file_path` has been provided
        in the constructor.

        """

        if self.file_path:
            with open(self.file_path, "w") as f:
                yaml.dump(self.root_node, f)

    def load(self, file_path: str) -> None:
        """Load log data from an input file.

        Args:
            file_path: File path to load data from.

        """

        with open(file_path, "r") as f:
            obj = yaml.load(f, Loader=yaml.Loader)
            self.stack = [obj]

    def close(self) -> None:
        """Close the logger."""

        self.save()

        for handler in self.logger.handlers:
            handler.flush()

    def _update_key(
        self,
        key: Any,
        value: Any,
        node: Optional[OrderedDict] = None,
        path: Optional[List[str]] = None,
        override_key: Optional[bool] = True,
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

    def _update(self, obj: Dict[str, Any], override_key: Optional[bool] = True) -> None:
        for k, v in obj.items():
            self._update_key(k, v, override_key=override_key)

    def log(
        self, obj: Union[Dict[str, Any], str], level: Optional[int] = None, override_key: Optional[bool] = True
    ) -> None:
        """Log the provided dictionary/string at the specified level.

        Args:
            obj: Object to log.
            level: Logging level.
            override_key: Whether key can be overridden if it's already in current node.

        """

        self.call_count += 1

        if isinstance(obj, dict):
            self._update(obj, override_key=override_key)
            message = ", ".join(f"{k}={v}" for k, v in obj.items())
        else:
            message = obj
            path = {
                logging.INFO: ["messages"],
                logging.DEBUG: ["debugs"],
                logging.WARNING: ["warnings"],
                logging.ERROR: ["errors"],
            }
            self._update_key(self.call_count, message, node=self.root_node, path=path[level], override_key=override_key)

        self.logger.log(msg=self.current_path + " " + message, level=level)

        if time.time() - self.timestamp > self.delay:
            self.save()
            self.timestamp = time.time()

    def info(self, obj: Union[Dict[str, Any], str], override_key: Optional[bool] = True) -> None:
        """Log the provided dictionary/string at the `info` level.

        Args:
            obj: Object to log.
            override_key: Whether key can be overridden if it's already in current node.

        """

        self.log(obj, level=logging.INFO, override_key=override_key)

    def debug(self, obj: Union[Dict[str, Any], str], override_key: Optional[bool] = True) -> None:
        """Log the provided dictionary/string at the `debug` level.

        Args:
            obj: Object to log.
            override_key: Whether key can be overridden if it's already in current node.

        """

        self.log(obj, level=logging.DEBUG, override_key=override_key)

    def warn(self, obj: Union[Dict[str, Any], str], override_key: Optional[bool] = True) -> None:
        """Log the provided dictionary/string at the `warning` level.

        Args:
            obj: Object to log.
            override_key: Whether key can be overridden if it's already in current node.

        """

        self.log(obj, level=logging.WARNING, override_key=override_key)

    def error(self, obj: Union[Dict[str, Any], str], override_key: Optional[bool] = True) -> None:
        """Log the provided dictionary/string at the `error` level.

        Args:
            obj: Object to log.
            override_key: Whether key can be overridden if it's already in current node.

        """

        self.log(obj, level=logging.ERROR, override_key=override_key)

    def pushd(self, *keys: Any) -> OrderedDictLogger:
        """Push the provided keys onto the current path stack.

        Returns:
            Instance of current logger.

        """

        self.paths.append([str(k) for k in keys])
        self.stack.append(None)  # Delays creation of node until it is needed

        return self  # Allows to call __enter__

    def popd(self) -> None:
        """Pop the last path and node off the stack."""

        if len(self.stack) == 1:
            self.warn("Invalid call. No available child in the stack.")
            return

        self.stack.pop()
        self.paths.pop()

    @staticmethod
    def set_global_instance(instance: OrderedDictLogger) -> None:
        """Set a global logger instance.

        Args:
            instance: Instance to be set globally.

        """

        global _logger
        _logger = instance

    @staticmethod
    def get_global_instance() -> OrderedDictLogger:
        """Get a global logger instance.

        Returns:
            Global logger.

        """

        global _logger
        return _logger


def get_global_logger() -> OrderedDictLogger:
    """Get a global logger instance.

    This method assures that if a global logger instance does not exist,
    it will be created and set as the global logger instance.

    Returns:
        Global logger.

    """

    try:
        logger = OrderedDictLogger.get_global_instance()
    except:
        OrderedDictLogger.set_global_instance(OrderedDictLogger(file_path="archai.log.yaml", delay=30.0))
        logger = OrderedDictLogger.get_global_instance()

    return logger
