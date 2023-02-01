# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from archai.discrete_search.evaluators.functional import EvaluationFunction


def test_evaluation_function():
    evaluator = EvaluationFunction(lambda a, d, b: 1)

    # Assert that evaluator can evaluate the argument function
    value = evaluator.evaluate(None, None, None)
    assert isinstance(evaluator.evaluation_fn, Callable)
    assert value == 1
