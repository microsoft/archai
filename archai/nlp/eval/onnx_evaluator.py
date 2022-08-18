# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-based manual evaluation.
"""

import copy
import math
import time
from typing import Any, Dict, Optional

import torch
from datasets.arrow_dataset import Dataset
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.data.data_collator import DataCollator

from archai.nlp import logging_utils
from archai.nlp.eval.metrics import lm_accuracy_metric

logger = logging_utils.get_logger(__name__)


def generate(onnx_session: InferenceSession, max_length: Optional[int] = 1, **inputs) -> torch.LongTensor:
    """Manually generates a set of tokens through greedy search.

    Args:
        onnx_session: An already instantiated ONNX-based session.
        max_length: Maximum amount of generated tokens.

    Returns:
        (torch.LongTensor): Tensor holding the input + predicted tokens.

    """

    for _ in range(max_length):
        # Gathers the inputs ready for ONNX
        input_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}

        # Passes through model and gathers the predictions
        logits = torch.from_numpy(onnx_session.run(["logits"], input_onnx)[0])
        sampled_logits = torch.argmax(logits[:, -1], dim=1).unsqueeze(-1)

        # Concatenates the predicted tokens with the inputs
        inputs["input_ids"] = torch.cat((inputs["input_ids"], sampled_logits), dim=-1)

    return inputs["input_ids"]


def manual_evaluate(
    onnx_session: InferenceSession,
    eval_dataset: Dataset,
    data_collator: Optional[DataCollator] = None,
    batch_size: Optional[int] = 1,
    n_seed_tokens: Optional[int] = 1,
    n_accuracy_type: Optional[int] = 1,
) -> Dict[str, Any]:
    """Manually evaluates an ONNX model.

    Args:
        onnx_session: An already instantiated ONNX-based session.
        eval_dataset: Evaluation (testing) dataset.
        data_collator: Collation function to be applied into data loader.
        batch_size: Size of batches.
        n_seed_tokens: Number of seed tokens.
        n_accuracy_type: Number of accuracies to be used, e.g., acc@1, acc@2, ..., acc@n.

    Returns:
        (Dict[str, Any]): A dictionary holding the evaluation metrics.

    """

    logger.info("Manually evaluating ONNX model ...")

    # Creates a DataLoader with receiving arguments
    data_loader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    # Defines initial evaluation metrics
    eval_loss = 0.0
    eval_acc = {f"acc@{i+1}": 0.0 for i in range(n_accuracy_type)}
    start_time = time.time()

    # Iterates through all samples
    for step, inputs in enumerate(tqdm(data_loader)):
        # Gathers a list of possible accuracies (acc@1, ..., acc@n)
        acc_types = [(n_seed_tokens, n_seed_tokens + i) for i in range(1, n_accuracy_type + 1)]

        for acc_type in acc_types:
            # Defines the name of current key
            key = f"acc@{acc_type[1]-acc_type[0]}"

            # Resizes both inputs and labels to correspond to accuracy type
            resized_inputs = copy.deepcopy(inputs)
            resized_inputs = {k: v[:, 0 : acc_type[0]] for k, v in resized_inputs.items()}
            resized_labels = copy.deepcopy(inputs["labels"])[:, acc_type[0] : acc_type[1]]

            # Generates a set of tokens
            outputs = generate(onnx_session, **resized_inputs, max_length=acc_type[1] - acc_type[0])
            outputs = outputs[:, acc_type[0] : acc_type[1]]

            # Calculates the accuracy
            eval_acc[key] += lm_accuracy_metric(outputs, resized_labels)

        # Performs the forward pass and calculates the evaluation loss
        input_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
        eval_loss += onnx_session.run(["loss"], input_onnx)[0]

    # Defines ending time and number of steps
    end_time = time.time()
    step += 1

    # Calculates final evaluation metrics
    eval_loss = (eval_loss / step).item()
    eval_acc = {k: (v / step).item() for k, v in eval_acc.items()}
    eval_runtime = end_time - start_time

    # Generates the output dictionary with some metrics
    output = {
        "eval_loss": eval_loss,
        "eval_ppl": math.exp(eval_loss),
        "eval_acc": eval_acc,
        "eval_runtime": eval_runtime,
        "eval_samples_per_second": len(eval_dataset) / eval_runtime,
        "eval_steps_per_second": step / eval_runtime,
        "step": step,
    }

    return output
