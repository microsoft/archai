# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable training arguments with huggingface/transformers.
"""

from dataclasses import dataclass, field

from transformers.training_args import TrainingArguments


@dataclass
class DistillerTrainingArguments(TrainingArguments):
    """Inherits from TrainingArguments and customizes distillation arguments."""

    alpha: float = field(default=0.5, metadata={"help": "Weight ratio between student and KD losses."})

    temperature: float = field(default=1.0, metadata={"help": "Annealing ratio for the softmax activations."})
