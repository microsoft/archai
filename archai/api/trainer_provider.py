# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from overrides import EnforceOverrides


class TrainerProvider(EnforceOverrides):
    """"""

    def __init__(self) -> None:
        """"""

        super().__init__()

    @abstractmethod
    def _training_step():
        """"""

    @abstractmethod
    def train():
        """"""

    @abstractmethod
    def _evaluation_step():
        """"""

    @abstractmethod
    def evaluate():
        """"""

    @abstractmethod
    def _prediction_step():
        """"""

    @abstractmethod
    def predict():
        """"""
