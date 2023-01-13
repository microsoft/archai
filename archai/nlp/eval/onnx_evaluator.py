# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-based manual evaluation."""

import copy
import math
import time
from typing import Any, Callable, Dict, Optional

import torch
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from archai.nlp.eval.eval_utils import lm_accuracy


def generate(onnx_session: InferenceSession, max_length: Optional[int] = 1, **inputs) -> torch.LongTensor:
    """Generate a sequence of tokens using greedy search with an ONNX model.

    This function generates a sequence of tokens by repeatedly running the provided
    ONNX session with the given inputs, and selecting the token with the highest
    predicted probability at each step. The process is repeated for the specified
    maximum number of steps.

    Args:
        onnx_session: An already instantiated ONNX-based session to use for prediction.
        max_length: Maximum number of tokens to generate.

    Returns:
        Tensor holding the input tokens plus the predicted tokens.

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
    data_collator: Optional[Callable] = None,
    batch_size: Optional[int] = 1,
    n_seed_tokens: Optional[int] = 1,
    n_accuracy_type: Optional[int] = 1,
) -> Dict[str, Any]:
    """Evaluate an ONNX model.

    Args:
        onnx_session: An already instantiated ONNX-based session to use for prediction.
        eval_dataset: Evaluation (testing) dataset.
        data_collator: Collation function to be applied to the data in the dataset.
        batch_size: Size of batches to use when iterating through the dataset.
        n_seed_tokens: Number of seed tokens to use when generating sequences.
        n_accuracy_type: Number of accuracies to calculate, ranging from "acc@1" to "acc@n".

    Returns:
        A dictionary containing the evaluation metrics and runtime statistics.

    """

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
            eval_acc[key] += lm_accuracy(outputs, resized_labels)["lm_accuracy"]

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
