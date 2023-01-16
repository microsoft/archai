# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from overrides import EnforceOverrides


class TrainerBase(EnforceOverrides):
    """Abstract class for trainer."""

    def __init__(self) -> None:
        """Initializes trainer."""

        pass

    @abstractmethod
    def train(self) -> None:
        """Trains a model.

        This function needs to be overriden as any logic can be applied to
        train a model.

        Examples:
            >>> return pytorch_lightining.trainer.Trainer().fit(model, train_dataloaders=train_dataloader)

        """

        pass

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluates a model.

        This function needs to be overriden as any logic can be applied to
        evaluate a model.

        Examples:
            >>> return pytorch_lightining.trainer.Trainer().test(model, dataloaders=val_dataloader)

        """

        pass

    @abstractmethod
    def predict(self) -> None:
        """Predicts with a model.

        This function needs to be overriden as any logic can be applied to
        predict with a model.

        Examples:
            >>> return pytorch_lightining.trainer.Trainer().predict(model, dataloaders=predict_dataloader)

        """

        pass
