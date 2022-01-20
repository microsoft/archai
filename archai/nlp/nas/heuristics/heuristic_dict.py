# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dictionary and enumerators that allows the implementation
and usage of Opytimizer-based heuristics.
"""

from enum import Enum
from typing import Dict


class HeuristicClassType(Enum):
    """An enumerator that defines the type of available classes to be loaded.

    """

    # Types of classes
    PACKAGE = 0
    HEURISTIC = 1


class HeuristicDict(Dict):
    """Dictionary that defines the type of available heuristics to be loaded.

    The order of classes must be asserted to the same defined by HeuristicClassType.

    """

    # Particle Swarm Optimization
    PSO = ('swarm', 'PSO')
