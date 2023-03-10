# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Any, Dict, Iterable, Optional, Union

import deepspeed
import mlflow
import torch
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from overrides import overrides
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from archai.api.trainer_base import TrainerBase
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.trainers.ds_training_args import DsTrainingArguments

logger = OrderedDictLogger(source=__name__)


def _create_deepspeed_config() -> Dict[str, Any]:
    return {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 8,
        "fp16": {"enabled": True, "initial_scale_power": 12},
        "zero_optimization": {"stage": 0},
        "optimizer": {"type": "AdamW", "params": {"lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8}},
    }


class DsTrainer(TrainerBase):
    """DeepSpeed trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: Optional[DsTrainingArguments] = None,
        optimizer: Optional[Optimizer] = None,
        model_parameters: Optional[Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]]] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
        mpu: Optional[Any] = None,
        dist_init_required: Optional[bool] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """Initialize by creating the DeepSpeed engine.

        Args:
            model: Model to be trained or evaluated.
            args: DeepSpeed training arguments. If not provided, a default instance
                of `DsTrainingArguments` will be used.
            optimizer: Optimizer to be used for training.
            model_parameters: Model parameters to be used for training.
            lr_scheduler: Learning rate scheduler to be used for training.
            mpu: Model parallelism unit.
            dist_init_required: Whether distributed initialization is required.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.

        """

        deepspeed.init_distributed()

        if args is None:
            args = DsTrainingArguments(_create_deepspeed_config())
        assert isinstance(args, DsTrainingArguments), "`args` should be an instance of `DsTrainingArguments`."
        self.args = args

        assert isinstance(
            model, torch.nn.Sequential
        ), "`model` should be an instance of `torch.nn.Sequential` for Pipeline Parallelism."
        model = PipelineModule(
            layers=model,
            num_stages=self.args.pipe_parallel_size,
            loss_fn=self.args.pipe_parallel_loss_fn,
            partition_method=self.args.pipe_parallel_partition_method,
            activation_checkpoint_interval=self.args.pipe_parallel_activation_checkpoint_steps,
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

        if self.engine.global_rank == 0:
            mlflow.start_run()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def _get_dataloader(
        self, dataset: Dataset, sampler: Optional[Sampler] = None, shuffle: Optional[bool] = False
    ) -> DataLoader:
        if sampler is None:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.engine.dp_world_size,
                rank=self.engine.mpu.get_data_parallel_rank(),
                shuffle=shuffle,
            )

        return DataLoader(dataset, sampler=sampler, drop_last=True, batch_size=self.engine.micro_batch_size)

    @overrides
    def train(self) -> None:
        """Train a model."""

        logger.info("Starting training ...")
        logger.debug(f"Training arguments: {self.args.to_dict()}")

        train_dataloader = self._get_dataloader(self.train_dataset, shuffle=True)
        train_iterator = iter(RepeatingLoader(train_dataloader))

        for step in range(self.args.max_steps):
            loss = self.engine.train_batch(data_iter=train_iterator)

            if self.engine.global_rank == 0:
                float_loss = loss.mean().item()

                mlflow.log_metric("train_loss", float_loss, step=step + 1)
                mlflow.log_metric("ppl", math.exp(float_loss), step=step + 1)

                do_periodic_logging = (step + 1) % self.args.logging_steps == 0
                if do_periodic_logging:
                    logger.info(f"Step: {step + 1} | Loss: {float_loss:.3f} | PPL: {math.exp(float_loss):.3f}")

            do_periodic_eval = (step + 1) % self.args.eval_steps == 0
            if do_periodic_eval and self.args.do_eval:
                assert self.eval_dataset, "`eval_dataset` must be supplied if `args.do_eval` is True."
                float_eval_loss = self.evaluate(self.eval_dataset)

                if self.engine.global_rank == 0:
                    mlflow.log_metric("eval_loss", float_eval_loss, step=step + 1)
                    mlflow.log_metric("eval_ppl", math.exp(float_eval_loss), step=step + 1)
                    logger.info(f"Eval Loss: {float_eval_loss:.3f} | Eval PPL: {math.exp(float_eval_loss):.3f}")

            do_periodic_checkpoint = (step + 1) % self.args.save_steps == 0
            if do_periodic_checkpoint:
                self.engine.save_checkpoint(self.args.output_dir, step + 1)

        if self.engine.global_rank == 0:
            mlflow.end_run()

    @overrides
    def evaluate(self, eval_dataset: Dataset) -> float:
        """Evaluate a model.

        Args:
            eval_dataset: Evaluation dataset.

        Returns:
            Evaluation loss.

        """

        eval_dataloader = self._get_dataloader(eval_dataset, shuffle=False)
        eval_iterator = iter(eval_dataloader)

        n_eval_steps = self.args.eval_max_steps or len(eval_dataloader)
        eval_loss = 0.0

        for _ in range(n_eval_steps):
            loss = self.engine.eval_batch(data_iter=eval_iterator)
            eval_loss += loss.mean().item()

        eval_loss / n_eval_steps

        return eval_loss

    @overrides
    def predict(self) -> None:
        """Predict with a model."""

        raise NotImplementedError
