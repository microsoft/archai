# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that are re-used across Text Predict.
"""

import re
from typing import Any, Dict, Optional

# Token-separation constants
WORD_TOKEN_SEPARATOR = "Ġ \nĊ\t\.;:,\'\"`<>\(\)\{\}\[\]\|\!@\#\$\%\^\&\*=\+\?/\\_\-~" # pylint: disable=anomalous-backslash-in-string
WORD_TOKEN_SEPARATOR_SET = set(WORD_TOKEN_SEPARATOR)
# TODO: FIX Tokenizer to work with this well
RE_SPLIT = re.compile('^(.*)([' + WORD_TOKEN_SEPARATOR + '].*)$', re.MULTILINE | re.DOTALL)
# WORD_TOKEN_SEPARATOR = 'Ġ \nĊ\t\.;:,\'\"`<>\(\)\{\}\[\]\|\!@\#\$\%\^\&\*=\+\?/\\_\-~'
# WORD_TOKEN_SEPARATOR_SET = set(WORD_TOKEN_SEPARATOR)
# RE_SPLIT = re.compile('^(.*)([' + WORD_TOKEN_SEPARATOR + '].*)$', re.MULTILINE | re.DOTALL)
MAX_LIST_LEN = 10


def get_settings(obj: Any, recursive: Optional[bool] = True) -> Dict[str, Any]:
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
