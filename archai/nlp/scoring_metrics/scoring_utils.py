# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scorer-based utilities, such as settings getter.
"""

import re

WORD_TOKEN_SEPARATOR = "Ġ \nĊ\t\.;:,\'\"`<>\(\)\{\}\[\]\|\!@\#\$\%\^\&\*=\+\?/\\_\-~" # pylint: disable=anomalous-backslash-in-string
WORD_TOKEN_SEPARATOR_SET = set(WORD_TOKEN_SEPARATOR)
RE_SPLIT = re.compile('^(.*)([' + WORD_TOKEN_SEPARATOR + '].*)$', re.MULTILINE | re.DOTALL)

def get_settings(obj, recursive=True):
    """Return object variables, for saving out as settings."""
    MAX_LIST_LEN = 10 # pylint: disable=invalid-name

    variables = {}
    var_dict = dict(vars(obj.__class__))
    try:
        var_dict.update(dict(vars(obj)))
    except TypeError:
        pass
    for k, v in var_dict.items():
        if k[0] == "_":
            continue

        if isinstance(v, (int, float, str)):
            variables[k.lower()] = v
        elif isinstance(v, list) and (len(v) == 0 or isinstance(v[0], (int, float, str))):
            variables[k.lower()] = v[:MAX_LIST_LEN]
        elif isinstance(v, set) and (len(v) == 0 or isinstance(next(iter(v)), (int, float, str))):
            variables[k.lower()] = list(v)[:MAX_LIST_LEN]
        elif recursive:
            settings_fn = getattr(v, "settings", None)
            if callable(settings_fn):
                variables[k.lower()] = settings_fn()

    return variables