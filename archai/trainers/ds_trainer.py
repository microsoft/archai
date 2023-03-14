# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import time
from typing import Any, Dict, Iterable, Optional, Tuple, Union

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

        self.client_state = {"step": 0}

    def _get_dataloader(
        self,
        dataset: Dataset,
        sampler: Optional[Sampler] = None,
        shuffle: Optional[bool] = False,
    ) -> DataLoader:
        if sampler is None:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.engine.dp_world_size,
                rank=self.engine.mpu.get_data_parallel_rank(),
                shuffle=shuffle,
            )

        return DataLoader(
            dataset,
            batch_size=self.engine.micro_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=True,
        )

    @overrides
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        resume_optimizer_state: Optional[bool] = True,
        resume_lr_scheduler_state: Optional[bool] = True,
    ) -> None:
        """Train a model."""

        logger.info("Starting training ...")
        logger.debug(f"Training arguments: {self.args.to_dict()}")

        current_step = 0
        if resume_from_checkpoint:
            logger.info(f"Loading from checkpoint: {resume_from_checkpoint}")
            try:
                _, self.client_state = self.engine.load_checkpoint(
                    resume_from_checkpoint,
                    load_optimizer_states=resume_optimizer_state,
                    load_lr_scheduler_states=resume_lr_scheduler_state,
                )
                current_step = self.client_state["step"]
            except:
                pass

        train_dataloader = self._get_dataloader(self.train_dataset, shuffle=True)
        train_iterator = iter(RepeatingLoader(train_dataloader))
        train_time = time.time()

        for step in range(current_step, self.args.max_steps):
            step_time = time.time()
            loss = self.engine.train_batch(data_iter=train_iterator)
            step_time = time.time() - step_time

            if self.engine.global_rank == 0:
                float_loss = loss.mean().item()
                samples_per_second = self.engine.train_batch_size() / step_time
                learning_rate = self.engine.get_lr()[0]

                mlflow.log_metrics(
                    {
                        "train/loss": float_loss,
                        "train/ppl": math.exp(float_loss),
                        "train/learning_rate": learning_rate,
                        "train/samples_per_second": samples_per_second,
                        "train/step_runtime": step_time,
                    },
                    step=step + 1,
                )

                do_periodic_logging = (step + 1) % self.args.logging_steps == 0
                if do_periodic_logging:
                    logger.info(
                        f"Step: {step + 1} | Time: {step_time:.3f} | "
                        + f"LR: {learning_rate} | Samples/s: {samples_per_second:.3f} | "
                        + f"Loss: {float_loss:.3f} | PPL: {math.exp(float_loss):.3f}"
                    )

            do_periodic_eval = (step + 1) % self.args.eval_steps == 0
            if do_periodic_eval and self.args.do_eval:
                assert self.eval_dataset, "`eval_dataset` must be supplied if `args.do_eval` is True."
                eval_loss, eval_time, eval_samples_per_second, eval_steps_per_second = self.evaluate(self.eval_dataset)

                if self.engine.global_rank == 0:
                    eval_idx = (step + 1) // self.args.eval_steps
                    mlflow.log_metrics(
                        {
                            "eval/loss": eval_loss,
                            "eval/ppl": math.exp(eval_loss),
                            "eval/runtime": eval_time,
                            "eval/samples_per_second": eval_samples_per_second,
                            "eval/steps_per_second": eval_steps_per_second,
                        },
                        step=eval_idx,
                    )
                    logger.info(
                        f"Eval: {eval_idx} | Time: {eval_time:.3f} | "
                        + f"Samples/s: {eval_samples_per_second:.3f} | Loss: {eval_loss:.3f} | "
                        + f"PPL: {math.exp(eval_loss):.3f}"
                    )

            do_periodic_checkpoint = (step + 1) % self.args.save_steps == 0
            if do_periodic_checkpoint:
                self.client_state["step"] = step + 1
                self.engine.save_checkpoint(self.args.output_dir, step + 1, client_state=self.client_state)

        train_time = time.time() - train_time

        if self.engine.global_rank == 0:
            mlflow.log_metric("train/time", train_time)
            mlflow.end_run()

    @overrides
    def evaluate(self, eval_dataset: Dataset) -> Tuple[float, float, float, float]:
        """Evaluate a model.

        Args:
            eval_dataset: Evaluation dataset.

        Returns:
            Evaluation loss, time, samples per second and steps per second.

        """

        eval_dataloader = self._get_dataloader(eval_dataset, shuffle=False)
        eval_iterator = iter(eval_dataloader)

        n_eval_steps = self.args.eval_max_steps or len(eval_dataloader)
        eval_loss, eval_time = 0.0, time.time()

        for _ in range(n_eval_steps):
            loss = self.engine.eval_batch(data_iter=eval_iterator)
            eval_loss += loss.mean().item()

        eval_loss /= n_eval_steps

        eval_time = time.time() - eval_time
        eval_samples_per_second = (n_eval_steps * self.engine.train_batch_size()) / eval_time
        eval_steps_per_second = n_eval_steps / eval_time

        return eval_loss, eval_time, eval_samples_per_second, eval_steps_per_second

    @overrides
    def predict(self) -> None:
        """Predict with a model."""

        raise NotImplementedError
