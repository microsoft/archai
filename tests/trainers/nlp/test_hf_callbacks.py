# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from unittest.mock import MagicMock

from transformers import TrainerControl, TrainerState, TrainingArguments

from archai.trainers.nlp.hf_callbacks import (
    BPCTrainerCallback,
    PerplexityTrainerCallback,
)


def test_bpc_trainer_callback():
    callback = BPCTrainerCallback()

    args = MagicMock(spec=TrainingArguments)
    state = MagicMock(spec=TrainerState)
    state.log_history = [{"loss": 0.5, "eval_loss": 0.4, "test_loss": 0.3}]
    control = MagicMock(spec=TrainerControl)

    # Assert that the bpc values were added to the log history
    callback.on_log(args, state, control)
    assert state.log_history[-1]["bpc"] == 0.5 / math.log(2)
    assert state.log_history[-1]["eval_bpc"] == 0.4 / math.log(2)

    # Assert that the bpc values were added to the metrics dictionary
    metrics = {"eval_loss": 0.25, "test_loss": 0.2}
    callback.on_evaluate(args, state, control, metrics)
    assert metrics["eval_bpc"] == 0.25 / math.log(2)
    assert metrics["test_bpc"] == 0.2 / math.log(2)


def test_perplexity_trainer_callback():
    callback = PerplexityTrainerCallback()

    args = MagicMock(spec=TrainingArguments)
    state = MagicMock(spec=TrainerState)
    state.log_history = [{"loss": 0.5, "eval_loss": 0.4, "test_loss": 0.3}]
    control = MagicMock(spec=TrainerControl)

    # Assert that the perplexity values were added to the log history
    callback.on_log(args, state, control)
    assert state.log_history[-1]["ppl"] == math.exp(0.5)
    assert state.log_history[-1]["eval_ppl"] == math.exp(0.4)

    # Assert that the perplxity values were added to the metrics dictionary
    metrics = {"eval_loss": 0.25, "test_loss": 0.2}
    callback.on_evaluate(args, state, control, metrics)
    assert metrics["eval_ppl"] == math.exp(0.25)
    assert metrics["test_ppl"] == math.exp(0.2)
