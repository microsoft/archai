# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Union

import deepspeed
import torch
from deepspeed.pipe import PipelineModule
from overrides import overrides

from archai.api.trainer_base import TrainerBase
from archai.trainers.ds_training_args import DsTrainingArguments


class DsTrainer(TrainerBase):
    """DeepSpeed trainer."""

    def __init__(
        self,
        model,
        args=None,
        optimizer=None,
        model_parameters=None,
        training_data=None,
        lr_scheduler=None,
        mpu=None,
        dist_init_required=None,
        collate_fn=None,
    ) -> None:
        """"""

        deepspeed.init_distributed()

        if args is None:
            args = DsTrainingArguments({})
        assert isinstance(args, DsTrainingArguments), "`args` should be an instance of `DsTrainingArguments`."
        self.args = args

        if self.args.pipeline_parallalelism:
            assert isinstance(
                model, torch.nn.Sequential
            ), "`model` should be an instance of `torch.nn.Sequential` for Pipeline Parallelism."
            model = PipelineModule(
                layers=model,
                num_stages=self.args.pp_size,
                loss_fn=self.args.pp_loss_fn,
                partition_method=self.args.pp_partition_method,
                activation_checkpoint_interval=self.args.pp_activation_checkpoint_interval,
            )

        self.engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters or [p for p in model.parameters() if p.requires_grad],
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config=self.args.deepspeed_config,
        )

    @overrides
    def train(
        self,
    ) -> None:
        for _ in range(self.args.max_steps):
            self.engine.train_batch()

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
