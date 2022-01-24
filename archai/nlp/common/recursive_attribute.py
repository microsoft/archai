# Copyright (c) https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
# Licensed under the MIT license.

"""Recursive attributes getters and setters.
"""

import functools
from typing import Any


def rsetattr(obj: Any, attr: Any, val: Any) -> None:
    """Recursively sets an attribute, i.e., works for multi-nested attributes.

    Args:
        obj: Object that holds the attribute.
        attr: Attribute to be set.
        val: Value to be set.
        
    """

    pre, _, post = attr.rpartition('.')

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


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
