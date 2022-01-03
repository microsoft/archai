# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scorer-based utilities, such as settings getter.
"""

from typing import Any, List, Optional


def get_settings(obj: Any,
                 recursive: Optional[bool] = True) -> List[Any]:
    """Gathers the object variables for saving them as settings.

    Args:
        obj: Input object.
        recursive: Whether gathering should be recursive or not.

    Returns:
        List[Any]: Settings based on input objects.
    
    """

    MAX_LIST_LEN = 10

    variables = {}
    var_dict = dict(vars(obj.__class__))

    try:
        var_dict.update(dict(vars(obj)))
    except TypeError:
        pass

    for k, v in var_dict.items():
        if k[0] == '_':
            continue

        if isinstance(v, (int, float, str)):
            variables[k.lower()] = v

        elif isinstance(v, list) and (len(v) == 0 or isinstance(v[0], (int, float, str))):
            variables[k.lower()] = v[:MAX_LIST_LEN]

        elif isinstance(v, set) and (len(v) == 0 or isinstance(next(iter(v)), (int, float, str))):
            variables[k.lower()] = list(v)[:MAX_LIST_LEN]

        elif recursive:
            settings_fn = getattr(v, 'settings', None)

            if callable(settings_fn):
                variables[k.lower()] = settings_fn()

    return variables
