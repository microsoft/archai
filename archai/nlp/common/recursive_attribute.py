# Copyright (c) https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
# Licensed under the MIT license.

"""Recursive attributes getters and setters.
"""

import functools
from typing import Any


def rsetattr(obj: Any, attr: Any, value: Any) -> None:
    """Recursively sets an attribute, i.e., works for multi-nested attributes.

    Args:
        obj: Object that holds the attribute.
        attr: Attribute to be set.
        value: Value to be set.
        
    """

    pre_attr, _, post_attr = attr.rpartition('.')

    return setattr(rgetattr(obj, pre_attr) if pre_attr else obj, post_attr, value)


def rgetattr(obj: Any, attr: Any, *args) -> Any:
    """Recursively gets an attribute, i.e., works for multi-nested attributes.

    Args:
        obj: Object that holds the attribute.
        attr: Attribute to be gathered.
    
    Returns:
        (Any): Recursively gathered attribute.
        
    """

    def _getattr(obj: Any, attr: Any) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
