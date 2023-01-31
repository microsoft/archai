# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
from typing import Any


def rgetattr(obj: Any, attr: str, *args) -> Any:
    """Recursively get an attribute from an object.

    This function allows accessing nested attributes by separating each level with a dot (e.g., "attr1.attr2.attr3").
    If any attribute along the chain does not exist, the function returns the default value
    specified in the `*args` parameter.

    Args:
        obj: Object from which the attribute will be retrieved.
        attr: Name of the attribute to be retrieved, with each level separated by a dot.

    Returns:
        Attribute from the object.

    Example:
        >>> obj = MyObject()
        >>> rgetattr(obj, "attr1.attr2.attr3")

    Reference:
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427

    """

    def _getattr(obj: Any, attr: Any) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively set an attribute on an object.

    This function allows setting nested attributes by separating each level with a dot (e.g., "attr1.attr2.attr3").

    Args:
        obj: Object on which the attribute will be set.
        attr: Name of the attribute to be set, with each level separated by a dot.
        value: New value for the attribute.

    Example:
        >>> obj = MyObject()
        >>> rsetattr(obj, "attr1.attr2.attr3", new_value)

    Reference:
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427

    """

    pre_attr, _, post_attr = attr.rpartition(".")

    return setattr(rgetattr(obj, pre_attr) if pre_attr else obj, post_attr, value)
