# Implementation taken from: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427

"""Recursive attributes getters and setters.
"""

import functools
from typing import Any


def rsetattr(obj: Any, attr: Any, val: Any) -> callable:
    """Recursively sets an attribute.

    Args:
        obj: Object that holds the attribute.
        attr: Attribute to be set.
        val: Value to be set.
    
    Returns:
        (callable): Recursive call of function.
        
    """

    pre, _, post = attr.rpartition('.')

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: Any, attr: Any, *args) -> callable:
    """Recursively gets an attribute.

    Args:
        obj: Object that holds the attribute.
        attr: Attribute to be gathered.
    
    Returns:
        (callable): Recursive call of function.
        
    """

    def _getattr(obj: Any, attr: Any) -> callable:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
