# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

class HfAccelerateTrainer:
    """
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
    ) -> None:
        """"""
        pass

    def get_dataloader(self):
        """"""
        pass

    def create_optimizer_and_scheduler(self):
        """"""
        pass

    def training_step(self):
        """"""
        pass

    def train(self):
        """"""
        pass

    def prediction_step(self):
        """"""
        pass

    def predict(self):
        """"""
        pass

    def evaluate(self):
        """"""
        pass
