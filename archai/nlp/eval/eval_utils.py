# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluation-related utilities.
"""

from typing import Dict, Optional

import numpy as np
import torch


def lm_accuracy(predictions: np.ndarray, references: np.ndarray) -> Dict[str, float]:
    """Calculates a language modeling-compatible accuracy.

    Args:
        predictions: Predictions.
        references: Ground-truth labels.

    Returns:
        (Dict[str, float]): Accuracy score for language modeling.

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
    """Calculates the fixed-length perplexity.

    Args:
        model: Pre-trained model.
        input_ids: Encoded data to be evaluated.
        max_length: Maximum length of sequences.
        stride: Sliding window size.

    Returns:
        (Dict[str, float]): Fixed-length perplexity.

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
