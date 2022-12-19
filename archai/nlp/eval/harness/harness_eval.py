# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-based evaluation."""

from typing import Any, Dict, Optional

from tqdm import tqdm

from archai.nlp.eval.harness.harness_model import HarnessModel
from archai.nlp.eval.harness.harness_task import HarnessTask


def evaluate(
    harness_model: HarnessModel,
    harness_task: HarnessTask,
    n_few_shot: Optional[int] = 0,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a harness-based model on a harness-based task.

    This function performs the evaluation of a harness-based model on a harness-based task
    using a few-shot evaluation approach.

    Args:
        harness_model: Harness-based model to be evaluated.
        harness_task: Harness-based task on which the model is evaluated.
        n_few_shot: Number of few-shot samples to be used for the evaluation.
        description: Additional description to be added to the few-shot context.

    Returns:
        Output configuration and evaluation metrics.

    """

    # Evaluation samples should be extracted from test set,
    # which if it is not available, will be taken from the validation set
    if harness_task.has_test_set:
        eval_set = harness_task.test_set
    elif harness_task.has_validation_set:
        eval_set = harness_task.validation_set
    else:
        raise RuntimeError("`harness_task` should either have `test_set` or `validation_set`.")

    for sample in tqdm(eval_set):
        # Creates the context based on the number of few-shot samples
        context = harness_task.create_context(sample, n_few_shot=n_few_shot, description=description)

        # Creates the sampling procedure calls and ensures they are encoded in a list
        calls = harness_task.create_sampling_calls(sample, context)
        if not isinstance(calls, (list, tuple)):
            calls = [calls]

        # Performs the sampling and process the outputs
        results = tuple(getattr(harness_model, call.call_name)(*call.args, **call.kwargs) for call in calls)
        harness_task.compute_results(sample, results)

    # Calculates the final metrics
    output = harness_task.config
    output["eval"] = harness_task.compute_metrics()
    output["eval"]["n_few_shot"] = n_few_shot

    return output
