# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Attributes utilities, such as mapping, gathering and setting methods.
"""

import functools
from typing import Any, Dict, List, Optional, Union


def map_to_list(variable: Union[int, float, List[Union[int, float]]], size: int) -> List[Union[int, float]]:
    """Maps variables to a fixed length list.

    Args:
        variable: Variable to be mapped.
        size: Size to be mapped.

    Returns:
        (List[Union[int, float]]): Mapped list with fixed length.

    """

    if isinstance(variable, List):
        size_diff = size - len(variable)

        if size_diff < 0:
            return variable[:size]
        elif size_diff == 0:
            return variable
        elif size_diff > 0:
            return variable + [variable[-1]] * size_diff

    return [variable] * size


def rgetkey(obj_list: List[Dict[str, Any]], key: Optional[str] = None) -> List[Any]:
    """Recursively gets a key from a list of dictionaries.

    Args:
        obj: List of dictionaries that will have key gathered.
        key: Name of key that will be gathered.

    Returns:
        (List[Any]): List of keys from inputted dictionaries.

    """

    return [obj[key] for obj in obj_list if key in obj]


def rsetattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively sets an attribute.

    Args:
        obj: Object that will have attribute set.
        attr: Name of attribute that will be set.
        value: New value for the attribute.

    """

    # Copyright @ https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
    pre_attr, _, post_attr = attr.rpartition(".")

    return setattr(rgetattr(obj, pre_attr) if pre_attr else obj, post_attr, value)


def rgetattr(obj: Any, attr: str, *args) -> Any:
    """Recursively gets an attribute.

    Args:
        obj: Object that will have attribute gathered.
        attr: Name of attribute that will be gathered.

    Returns:
        (Any): Attribute from object.

    """

    # Copyright @ https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
    def _getattr(obj: Any, attr: Any) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class cached_property(property):
    """Mimics the @property decorator but caches the output."""

    def __get__(self, obj: Any, obj_type: Optional[Any] = None) -> Any:
        """Returns either an object or its cached version.

        Args:
            obj: Object to be returned.
            obj_type: Optional argument for compatibility.

        Returns:
            (Any): Object or its cached version.

        """

        if obj is None:
            return self

        if self.fget is None:
            raise AttributeError("Error when loading attribute")

        attr = "__cached_" + self.fget.__name__

        cached_obj = getattr(obj, attr, None)
        if cached_obj is None:
            cached_obj = self.fget(obj)
            setattr(obj, attr, cached_obj)

        return cached_obj
