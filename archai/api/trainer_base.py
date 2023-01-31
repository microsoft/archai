# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from overrides import EnforceOverrides


class TrainerBase(EnforceOverrides):
    """Abstract class for trainers.

    The `TrainerBase` class provides an abstract interface for training a model. The user
    is required to implement the `train`, `evaluate`, and `predict` methods. The `train` method
    should contain the logic for training the model, the `evaluate` method should contain
    the logic for evaluating the model, and the `predict` method should contain the logic
    for making predictions with the model.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    Examples:
        >>> class MyTrainer(TrainerBase):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>
        >>>     @overrides
        >>>     def train(self) -> None:
        >>>         return pytorch_lightining.trainer.Trainer().fit(model, train_dataloaders=train_dataloader)
        >>>
        >>>     @overrides
        >>>     def evaluate(self) -> None:
        >>>         return pytorch_lightining.trainer.Trainer().test(model, dataloaders=val_dataloader)
        >>>
        >>>     @overrides
        >>>     def predict(self) -> None:
        >>>         return pytorch_lightining.trainer.Trainer().predict(model, dataloaders=predict_dataloader)

    """

    def __init__(self) -> None:
        """Initialize the trainer."""

        pass

    @abstractmethod
    def train(self) -> None:
        """Train a model.

        This method should contain the logic for training the model.

        """

        pass

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluate a model.

        This method should contain the logic for evaluating the model.

        """

        pass

    @abstractmethod
    def predict(self) -> None:
        """Predict with a model.

        This method should contain the logic for making predictions with the model.

        """

        pass
