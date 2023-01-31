# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
from unittest.mock import MagicMock

from overrides import overrides

from archai.api.trainer_base import TrainerBase


class MyTrainer(TrainerBase):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def train(self) -> Any:
        return MagicMock()

    @overrides
    def evaluate(self) -> Any:
        return MagicMock()

    @overrides
    def predict(self) -> Any:
        return MagicMock()


def test_my_trainer():
    trainer = MyTrainer()

    assert trainer.train()
    assert trainer.evaluate()
    assert trainer.predict()
