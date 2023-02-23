# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Union

import deepspeed
from overrides import overrides

from archai.api.trainer_base import TrainerBase


class DsTrainer(TrainerBase):
    """DeepSpeed trainer."""

    def __init__(
        self,
        args=None,
        model=None,
        optimizer=None,
        model_parameters=None,
        training_data=None,
        lr_scheduler=None,
        mpu=None,
        init_required=None,
        collate_fn=None,
        config=None,
        config_params=None,
    ) -> None:
        """"""

        self.engine = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            init_required=init_required,
            collate_fn=collate_fn,
            config=config,
            config_params=config_params,
        )

    @overrides
    def train(
        self,
    ) -> None:
        pass

    @overrides
    def evaluate(
        self,
    ) -> None:
        pass

    @overrides
    def predict(
        self,
    ) -> None:
        pass
