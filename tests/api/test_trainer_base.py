# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

from overrides import overrides

from archai.api.trainer_base import TrainerBase


class MyTrainer(TrainerBase):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def train(self) -> None:
        return MagicMock()

    @overrides
    def evaluate(self) -> None:
        return MagicMock()

    @overrides
    def predict(self) -> None:
        return MagicMock()


def test_trainer():
    trainer = MyTrainer()

    # Assert that mocked methods run
    assert trainer.train()
    assert trainer.evaluate()
    assert trainer.predict()
