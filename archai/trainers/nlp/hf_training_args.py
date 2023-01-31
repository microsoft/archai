# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field

from transformers.training_args import TrainingArguments


@dataclass
class DistillerTrainingArguments(TrainingArguments):
    """Training arguments for distillation-based training.

    This class extends `TrainingArguments` and provides additional arguments
    specific to distillation-based training.

    Args:
        alpha: Weight ratio between the student and KD losses. This should be
            a value in the range [0, 1].
        temperature: Annealing ratio for the softmax activations. This value
            should be greater than 0.

    """

    alpha: float = field(default=0.5, metadata={"help": "Weight ratio between student and KD losses."})

    temperature: float = field(default=1.0, metadata={"help": "Annealing ratio for the softmax activations."})
