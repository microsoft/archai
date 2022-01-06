# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections.abc import Sized
from typing import Any, List, Union


def map_to_list(p: Union[Sized, Any], n: int) -> List[Any]:
    """Maps variables to lists with fixed lengths.

    Args:
        p: Variable to be mapped.
        n: Size of list to be mapped.

    """

    if isinstance(p, Sized):
        if len(p) == 1:
            return p * n
        return p
    return [p] * n
