# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from random import Random

import pytest

from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.evaluators.functional import EvaluationFunction


@pytest.fixture
def search_objectives():
    rng1 = Random(1)
    rng2 = Random(2)
    rng3 = Random(3)

    o1 = EvaluationFunction(lambda m, d, b: rng1.random())
    o2 = EvaluationFunction(lambda m, d, b: rng2.random())
    r = EvaluationFunction(lambda m, d, b: rng3.random())

    so = SearchObjectives()
    so.add_objective("Random1", o1, higher_is_better=False, compute_intensive=False, constraint=(0.0, 0.4))
    so.add_objective("Random2", o2, higher_is_better=True)

    so.add_constraint("Random3 constraint", r, constraint=(0.0, 0.6))

    return so
