# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Optional, Union

import deepspeed
import torch
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from overrides import overrides
from torch.utils.data import DataLoader

from archai.api.trainer_base import TrainerBase
from archai.trainers.ds_training_args import DsTrainingArguments


def _create_deepspeed_config() -> Dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": 256,
        "gradient_accumulation_steps": 16,
        "fp16": {"enabled": True, "initial_scale_power": 12},
        "zero_optimization": {"stage": 1, "reduce_bucket_size": 5e8},
        "optimizer": {"type": "Adam", "params": {"lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8}},
    }


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
        train_dataset=None,
        eval_dataset=None,
    ) -> None:
        """"""

        deepspeed.init_distributed()

        if args is None:
            args = DsTrainingArguments(_create_deepspeed_config())
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
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=dist_init_required,
            config=self.args.config,
        )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def _get_dataloader(self, dataset, sampler=None, shuffle=False):
        if sampler is None:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.engine.dp_world_size,
                rank=self.engine.mpu.get_data_parallel_rank(),
                shuffle=shuffle,
            )

        dataloader = DataLoader(dataset, sampler=sampler, drop_last=True, batch_size=self.engine.micro_batch_size)

        return iter(RepeatingLoader(dataloader))

    @overrides
    def train(
        self,
    ) -> None:
        train_dataloader = self._get_dataloader(self.train_dataset, shuffle=True)
        for step in range(self.args.max_steps):
            _ = self.engine.train_batch(train_dataloader)

            if step % self.args.eval_interval and self.args.do_eval:
                eval_dataloader = self._get_dataloader(self.eval_dataset, shuffle=False)
                self._evaluation_step(eval_dataloader)

            if step % self.args.save_steps:
                self.engine.save_checkpoint(self.args.output_dir, step)

    def _evaluation_step(self, eval_dataloader):
        eval_loss = 0.0

        for _ in range(len(eval_dataloader)):
            loss = self.engine.eval_batch(eval_dataloader)
            eval_loss += loss.mean().item()

        eval_loss / len(eval_dataloader)

        return eval_loss

    @overrides
    def evaluate(
        self,
        eval_dataset=None,
    ) -> None:
        eval_dataset = eval_dataset or self.eval_dataset
        assert eval_dataset, "`eval_dataset` must be supplied in constructor or evaluate()."

        eval_dataloader = self._get_dataloader(eval_dataset, shuffle=False)
        _ = self._evaluation_step(eval_dataloader)

    @overrides
    def predict(
        self,
    ) -> None:
        pass
