# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that handles mapping between types.
"""

from collections.abc import Sized
from typing import Any, Union


def map_to_list(p: Union[Sized, Any], n: int) -> Sized[Any]:
    """Maps variables to lists with fixed lengths.

    Args:
        p: Variable to be mapped.
        n: Size of list to be mapped.

    Returns:
        (Sized[Any]): List with a fixed length.

    """

    if isinstance(p, Sized):
        if len(p) == 1:
            return p * n
        return p
    return [p] * n
