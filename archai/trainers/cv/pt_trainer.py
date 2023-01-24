# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.api.trainer_base import TrainerBase


class TorchTrainer(TrainerBase):
    """PyTorch trainer."""

    def __init__(self) -> None:
        """Initialize the trainer."""

        super().__init__()

    @overrides
    def train(self) -> None:
        return super().train()

    @overrides
    def evaluate(self) -> None:
        return super().evaluate()

    @overrides
    def predict(self) -> None:
        return super().predict()
