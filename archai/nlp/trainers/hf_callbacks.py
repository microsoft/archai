# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable callbacks with huggingface/transformers.
"""

import math
from typing import Dict, Optional

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class BPCTrainerCallback(TrainerCallback):
    """Inherits from TrainerCallback and allows the output of bits per character values."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        super().__init__(*args, **kwargs)

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """Provides a customizable flow when a log event is called.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.

        """

        current_log = state.log_history[-1]

        # Checks whether last log comes from training step
        if "loss" in current_log:
            try:
                current_log["bpc"] = current_log["loss"] / math.log(2)
            except OverflowError:
                current_log["bpc"] = math.inf

        # Checks whether last log comes from evaluation step
        if "eval_loss" in current_log:
            try:
                current_log["eval_bpc"] = current_log["eval_loss"] / math.log(2)
            except OverflowError:
                current_log["eval_bpc"] = math.inf

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """Provides a customizable flow when an evaluate event is called.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            metrics: Evaluation metrics.

        """

        # Checks whether metrics have validation loss
        if "eval_loss" in metrics:
            try:
                metrics["eval_bpc"] = metrics["eval_loss"] / math.log(2)
            except OverflowError:
                metrics["eval_bpc"] = math.inf

        # Checks whether metrics have testing loss
        if "test_loss" in metrics:
            try:
                metrics["test_bpc"] = metrics["test_loss"] / math.log(2)
            except OverflowError:
                metrics["test_bpc"] = math.inf


class PerplexityTrainerCallback(TrainerCallback):
    """Inherits from TrainerCallback and allows the output of perplexity values."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        super().__init__(*args, **kwargs)

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """Provides a customizable flow when a log event is called.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.

        """

        current_log = state.log_history[-1]

        # Checks whether last log comes from training step
        if "loss" in current_log:
            try:
                current_log["ppl"] = math.exp(current_log["loss"])
            except OverflowError:
                current_log["ppl"] = math.inf

        # Checks whether last log comes from evaluation step
        if "eval_loss" in current_log:
            try:
                current_log["eval_ppl"] = math.exp(current_log["eval_loss"])
            except OverflowError:
                current_log["eval_ppl"] = math.inf

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """Provides a customizable flow when an evaluate event is called.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            metrics: Evaluation metrics.

        """

        # Checks whether metrics have validation loss
        if "eval_loss" in metrics:
            try:
                metrics["eval_ppl"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["eval_ppl"] = math.inf

        # Checks whether metrics have testing loss
        if "test_loss" in metrics:
            try:
                metrics["test_ppl"] = math.exp(metrics["test_loss"])
            except OverflowError:
                metrics["test_ppl"] = math.inf