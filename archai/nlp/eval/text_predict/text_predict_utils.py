# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Text Predict-based utilities, such as caching mechanism.
"""

from collections import OrderedDict
from typing import Any, Optional


class LRUCache(OrderedDict):
    """Implements functools.lru_cache() with an OrderedDict to avoid
    unhashable results.

    """

    def __init__(self, maxsize: Optional[int] = 128) -> None:
        """Overrides initialization method with custom arguments.

        Args:
            maxsize: Maximum size of the cache.

        """

        super().__init__()

        self.maxsize = maxsize

    def __getitem__(self, key: str) -> Any:
        """Retrieves a value from the cache.

        Args:
            key: Key to retrieve the value.

        Returns:
            (Any): Value.

        """

        value = super().__getitem__(key)
        self.move_to_end(key)

        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets a value in the cache.

        Args:
            key: Key to set the value.
            value: Value to be set.

        """

        if key in self:
            self.move_to_end(key)

        super().__setitem__(key, value)

        if len(self) > self.maxsize:
            old_key = next(iter(self))
            del self[old_key]
