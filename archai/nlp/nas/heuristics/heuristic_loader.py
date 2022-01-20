# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Loading utilities to import heuristics on demand.
"""

from importlib import import_module
from typing import Any

from archai.nlp.nas.heuristics.heuristic_dict import HeuristicClassType, HeuristicDict

# Path to the `optimizers` package
PACKAGE_PATH = 'opytimizer.optimizers'


def load_heuristic_from_args(heuristic_type: str, **kwargs) -> Any:
    """Performs the loading of a pre-defined heuristic and its
        corresponding class.

    Args:
        heuristic_type: Type of heuristic to be loaded.

    Returns:
        (Any): An instance of the loaded class.

    """
    
    # Gathers the index of corresponding package and heuristic
    cls_package_idx = HeuristicClassType.PACKAGE.value
    cls_heuristic_idx = HeuristicClassType.HEURISTIC.value

    # Gathers the available tuple to be loaded,
    # and its corresponding package and heuristic names
    cls_tuple = getattr(HeuristicDict, heuristic_type.upper())
    cls_package = cls_tuple[cls_package_idx]
    cls_heuristic = cls_tuple[cls_heuristic_idx]

    # Attempts to load the class
    try:
        cls_module = import_module(f'.{cls_package}.{heuristic_type}', PACKAGE_PATH)
        cls_instance = getattr(cls_module, cls_heuristic)
    except:
        raise ModuleNotFoundError

    # Initializes the class
    instance = cls_instance(**kwargs)

    return instance
