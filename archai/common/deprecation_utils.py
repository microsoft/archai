# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import warnings
from functools import wraps
from typing import Any, Callable, Optional

_deprecate_warnings_set = set()


def deprecated(
    message: Optional[str] = None, deprecate_version: Optional[str] = None, remove_version: Optional[str] = None
) -> None:
    """Decorator to mark a function or class as deprecated.

    Args:
        message: Message to include in the warning.
        deprecated_version: Version in which the function was deprecated.
            If `None`, the version will not be included in the warning message.
        remove_version: Version in which the function will be removed.
            If `None`, the version will not be included in the warning message.

    """

    def _deprecated(class_or_func: Callable) -> Callable:
        global _deprecate_warnings_set

        obj = class_or_func
        if inspect.isclass(class_or_func):
            obj = obj.__init__
        obj_name = class_or_func.__name__

        # Spaces are positioned with the intention of aligning
        # the message with the warning message
        dpr_version_message = f"in v{deprecate_version} " if deprecate_version else ""
        remove_version_message = f" in v{remove_version}" if remove_version else ""

        dpr_message = (
            f"`{obj_name}` has been deprecated {dpr_version_message}and will be removed{remove_version_message}."
        )
        dpr_message += f" {message}" if message else ""

        @wraps(obj)
        def __deprecated(*args, **kwargs) -> Any:
            # Avoids printing the same warning multiple times`
            obj_hash = hash(obj)
            if obj_hash not in _deprecate_warnings_set:
                warnings.warn(dpr_message, category=FutureWarning, stacklevel=2)
                _deprecate_warnings_set.add(obj_hash)

            return obj(*args, **kwargs)

        __deprecated._decorator_name_ = "deprecated"

        if inspect.isclass(class_or_func):
            class_or_func.__init__ = __deprecated
            return class_or_func

        return __deprecated

    return _deprecated
