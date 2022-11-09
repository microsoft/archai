# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Quantization-related utilities.
"""

import functools
from typing import Any


def rgetattr(obj: Any, attr: str, *args) -> Any:
    """Recursively gets an attribute.

    Reference:
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427

    Args:
        obj: Object that will have attribute gathered.
        attr: Name of attribute that will be gathered.

    Returns:
        (Any): Attribute from object.

    """

    def _getattr(obj: Any, attr: Any) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively sets an attribute.

    Reference:
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427

    Args:
        obj: Object that will have attribute set.
        attr: Name of attribute that will be set.
        value: New value for the attribute.

    """

    pre_attr, _, post_attr = attr.rpartition(".")

    return setattr(rgetattr(obj, pre_attr) if pre_attr else obj, post_attr, value)
