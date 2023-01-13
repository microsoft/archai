# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluation-related utilities."""

from typing import Any, Dict, Optional

import numpy as np
import torch


def lm_accuracy(predictions: np.ndarray, references: np.ndarray) -> Dict[str, float]:
    """Calculate language modeling accuracy.

    This function calculates the accuracy score for language modeling by comparing
    the predictions and ground-truth labels. Any predictions or references with a
    value of -100 are ignored.

    Args:
        predictions: 1D array of predictions.
        references: 1D array of ground-truth labels.

    Returns:
        Dictionary with a single key, "lm_accuracy", and the corresponding accuracy
            score as a float.

    """

    valid_mask = references != -100
    n_valid_references = valid_mask.sum()

    valid_predictions = predictions[valid_mask]
    valid_references = references[valid_mask]

    lm_accuracy = (valid_predictions == valid_references).sum() / n_valid_references

    return {"lm_accuracy": lm_accuracy}


def fixed_length_perplexity(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_length: Optional[int] = 512,
    stride: Optional[int] = 512,
) -> Dict[str, float]:
    """Calculate the fixed-length perplexity of a pre-trained model on a given dataset.

    This function calculates the perplexity of the given model on the input data,
    using a sliding window approach with the specified maximum length and stride.
    Any input tokens that fall outside the current window are masked with a value of -100.

    Args:
        model: Pre-trained model to evaluate.
        input_ids: Encoded input data to evaluate the model on.
        max_length: Maximum length of sequences to consider.
        stride: Sliding window size.

    Returns:
        Dictionary with a single key, "fixed_length_perplexity", and the
            corresponding perplexity score as a float.

    """

    model.eval()

    log_likelihood_list = []

    for i in range(0, input_ids.size(1), stride):
        start = max(i + stride - max_length, 0)
        end = min(i + stride, input_ids.size(1))

        input_ids_strided = input_ids[:, start:end]

        target_length = end - i
        target_ids_strided = input_ids_strided.clone()
        target_ids_strided[:, :-target_length] = -100

        with torch.no_grad():
            outputs = model(input_ids_strided, labels=target_ids_strided)

            log_likelihood = outputs[0] * target_length
            log_likelihood_list.append(log_likelihood)

    fixed_length_ppl = torch.exp(torch.stack(log_likelihood_list).sum() / end)

    return {"fixed_length_perplexity": fixed_length_ppl}


class cached_property(property):
    """Decorator that caches the output of the decorated function.

    This class is a subclass of the built-in `property` class, and can be used to
    decorate a function in a class in the same way as the `@property` decorator.
    The output of the decorated function is cached on the first call,
    and subsequent calls will return the cached value rather than re-running the function.

    """

    def __get__(self, obj: Any, obj_type: Optional[Any] = None) -> Any:
        """Return the value of the decorated function, or its cached version.

        Args:
            obj: Object that the decorated function belongs to.
            obj_type: Optional argument for compatibility.

        Returns:
            Value of the decorated function, or its cached version.

        """

        if obj is None:
            return self

        if self.fget is None:
            raise AttributeError("Error when loading attribute")

        attr = "__cached_" + self.fget.__name__

        cached_obj = getattr(obj, attr, None)
        if cached_obj is None:
            cached_obj = self.fget(obj)
            setattr(obj, attr, cached_obj)

        return cached_obj
