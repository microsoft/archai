# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that handles mapping between types.
"""

from typing import Union, Sized


def map_to_list(variable: Union[int, float, Sized], size: int) -> Sized:
    """Maps variables to lists with fixed lengths.

    Args:
        variable: Variable to be mapped.
        size: Size of list to be mapped.

    Returns:
        (Sized): List with a fixed length.

    """

    if isinstance(variable, Sized):
        size_diff = size - len(variable)

        if size_diff < 0:
            return variable[:size]
        elif size_diff == 0:
            return variable
        elif size_diff > 0:
            return variable + [variable[0]] * size_diff

    return [variable] * size
