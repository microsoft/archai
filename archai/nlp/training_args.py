# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable training arguments from huggingface/transformers.
"""

from typing import Any, Optional

from transformers.training_args import TrainingArguments


class ArchaiTrainingArguments(TrainingArguments):
    """Inherits from TrainingArguments and allows to be customized."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        super().__init__(*args, **kwargs)

    def get(self, key: str, default: Any) -> Any:
        """Creates a getter to mimic a dictionary's functionality.

        Args:
            key: Key to be gathered.
            default: Default value if key is not found.

        Returns:
            (Any): Attribute from class.

        """

        return getattr(self, key, default)


class ArchaiDistillerTrainingArguments(ArchaiTrainingArguments):
    """Inherits from ArchaiTrainingArguments and customizes distillation arguments."""

    def __init__(self, *args, alpha: Optional[float] = 0.5, temperature: Optional[float] = 1.0, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments.

        Args:
            alpha: Weight ratio between student and KD losses.
            temperature: Annealing ratio for the softmax activations.

        """

        super().__init__(*args, **kwargs)

        # Knowledge distillation attributes
        self.alpha = alpha
        self.temperature = temperature
