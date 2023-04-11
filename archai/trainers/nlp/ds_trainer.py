# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import math
import os
import time
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

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
from archai.trainers.nlp.ds_training_args import DsTrainingArguments

logger = OrderedDictLogger(source=__name__)


def _create_base_config() -> Dict[str, Any]:
    return {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 2,
        "fp16": {"enabled": True, "initial_scale_power": 12},
        "zero_optimization": {"stage": 0},
        "optimizer": {"type": "AdamW", "params": {"lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8}},
    }


class StatefulDistributedSampler(DistributedSampler):
    """Distributed sampler that supports resuming from a given step."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False,
        total_consumed_samples: Optional[int] = 0,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset: Dataset to be sampled.
            num_replicas: Number of replicas.
            rank: Rank of the current process.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed.
            drop_last: Whether to drop the last batch if it is smaller than the batch size.
            total_consumed_samples: Total number of samples consumed.

        """

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

        self.total_consumed_samples = total_consumed_samples

    def __iter__(self) -> Iterator:
        indices = list(super().__iter__())
        return iter(indices[((self.total_consumed_samples // self.num_replicas) % self.num_samples) :])


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
            args = DsTrainingArguments("tmp", ds_config=_create_base_config())
        assert isinstance(args, DsTrainingArguments), "`args` should be an instance of `DsTrainingArguments`."
        self.args = args

        if self.args.pipe_parallel_size > 0:
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
            config=self.args.ds_config,
        )

        if self.engine.global_rank == 0:
            mlflow.start_run()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.client_state = {"global_step": 0, "total_consumed_samples": 0, "log_history": []}

    @property
    def data_parallel_world_size(self) -> int:
        """Return the data parallel world size."""

        if self.engine.mpu:
            return self.engine.mpu.get_data_parallel_world_size()
        return None

    @property
    def data_parallel_rank(self) -> int:
        """Return the data parallel rank of the current process."""

        if self.engine.mpu:
            return self.engine.mpu.get_data_parallel_rank()
        return None

    def _get_dataloader(
        self,
        dataset: Dataset,
        sampler: Optional[Sampler] = None,
        shuffle: Optional[bool] = False,
        total_consumed_samples: Optional[int] = 0,
    ) -> DataLoader:
        if sampler is None:
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=self.data_parallel_world_size,
                rank=self.data_parallel_rank,
                shuffle=shuffle,
                total_consumed_samples=total_consumed_samples,
            )

        return DataLoader(
            dataset,
            batch_size=self.engine.train_micro_batch_size_per_gpu(),
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=True,
        )

    def train_batch_without_pipe_parallel(self, data_iter: Optional[Iterator] = None) -> torch.Tensor:
        """Train a batch without pipeline parallelism.

        Args:
            data_iter: Data iterator.

        Returns:
            Loss tensor.

        """

        gradient_accumulation_steps = self.engine.gradient_accumulation_steps()
        total_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            input_ids, _ = next(data_iter)
            input_ids = input_ids.to(self.engine.device)

            outputs = self.engine(input_ids, labels=input_ids)
            loss = outputs[0].mean()

            self.engine.backward(loss)
            self.engine.step()

            total_loss += loss

        return total_loss / gradient_accumulation_steps

    def eval_batch_without_pipe_parallel(self, data_iter: Optional[Iterator] = None) -> torch.Tensor:
        """Evaluate a batch without pipeline parallelism.

        Args:
            data_iter: Data iterator.

        Returns:
            Loss tensor.

        """

        with torch.no_grad():
            gradient_accumulation_steps = self.engine.gradient_accumulation_steps()
            total_loss = 0.0

            for _ in range(gradient_accumulation_steps):
                input_ids, _ = next(data_iter)
                input_ids = input_ids.to(self.engine.device)

                outputs = self.engine(input_ids, labels=input_ids)
                loss = outputs[0].mean()

                total_loss += loss

        return total_loss / gradient_accumulation_steps

    @overrides
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        resume_optimizer_state: Optional[bool] = True,
        resume_lr_scheduler_state: Optional[bool] = True,
    ) -> None:
        """Train a model.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume training from.
            resume_optimizer_state: Whether to resume optimizer state.
            resume_lr_scheduler_state: Whether to resume learning rate scheduler state.

        """

        logger.info("Starting training ...")
        logger.debug(f"Training arguments: {self.args.to_dict()}")

        global_step = 0
        total_consumed_samples = 0
        log_history = []

        if resume_from_checkpoint:
            logger.info(f"Loading from checkpoint: {resume_from_checkpoint}")
            try:
                _, self.client_state = self.engine.load_checkpoint(
                    resume_from_checkpoint,
                    load_optimizer_states=resume_optimizer_state,
                    load_lr_scheduler_states=resume_lr_scheduler_state,
                )
                global_step = self.client_state["global_step"]
                total_consumed_samples = self.client_state["total_consumed_samples"]
                log_history = self.client_state["log_history"]
            except:
                pass

        train_dataloader = self._get_dataloader(
            self.train_dataset,
            shuffle=True,
            total_consumed_samples=total_consumed_samples,
        )
        train_iterator = iter(RepeatingLoader(train_dataloader))
        train_time = time.time()

        for step in range(global_step, self.args.max_steps):
            step_time = time.time()

            if self.args.pipe_parallel_size > 0:
                loss = self.engine.train_batch(data_iter=train_iterator)
            else:
                loss = self.train_batch_without_pipe_parallel(data_iter=train_iterator)

            step_time = time.time() - step_time

            if self.engine.global_rank == 0:
                float_loss = loss.mean().item()
                samples_per_second = self.engine.train_batch_size() / step_time
                learning_rate = self.engine.get_lr()[0]

                metrics = {
                    "train/step": step + 1,
                    "train/loss": float_loss,
                    "train/ppl": math.exp(float_loss),
                    "train/learning_rate": learning_rate,
                    "train/samples_per_second": samples_per_second,
                    "train/step_runtime": step_time,
                }

                log_history.append(metrics)
                mlflow.log_metrics(metrics, step=step + 1)

                do_periodic_logging = (step + 1) % self.args.logging_steps == 0
                if do_periodic_logging:
                    logger.info(
                        f"Step: {step + 1} | LR: {learning_rate} | "
                        + f"Loss: {float_loss:.3f} | Samples/s: {samples_per_second:.3f} | "
                        + f"PPL: {math.exp(float_loss):.3f}"
                    )

            do_periodic_eval = (step + 1) % self.args.eval_steps == 0
            if do_periodic_eval and self.args.do_eval:
                assert self.eval_dataset, "`eval_dataset` must be supplied if `args.do_eval` is True."
                eval_loss, eval_time, eval_samples_per_second, eval_steps_per_second = self.evaluate(self.eval_dataset)

                if self.engine.global_rank == 0:
                    eval_idx = (step + 1) // self.args.eval_steps
                    metrics = {
                        "eval/idx": eval_idx,
                        "eval/loss": eval_loss,
                        "eval/ppl": math.exp(eval_loss),
                        "eval/runtime": eval_time,
                        "eval/samples_per_second": eval_samples_per_second,
                        "eval/steps_per_second": eval_steps_per_second,
                    }

                    log_history.append(metrics)
                    mlflow.log_metrics(metrics, step=eval_idx)

                    logger.info(
                        f"Eval: {eval_idx} | Seconds: {eval_time:.3f} | "
                        + f"Samples/s: {eval_samples_per_second:.3f} | Loss: {eval_loss:.3f} | "
                        + f"PPL: {math.exp(eval_loss):.3f}"
                    )

            do_periodic_checkpoint = (step + 1) % self.args.save_steps == 0
            if do_periodic_checkpoint:
                self.client_state["global_step"] = step + 1
                self.client_state["total_consumed_samples"] = self.engine.global_samples
                self.client_state["log_history"] = log_history

                self.engine.save_checkpoint(self.args.output_dir, step + 1, client_state=self.client_state)
                with open(os.path.join(self.args.output_dir, "trainer_state.json"), "w") as f:
                    json.dump(self.client_state, f)

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
        eval_iterator = iter(RepeatingLoader(eval_dataloader))

        n_eval_steps = self.args.eval_max_steps or len(eval_dataloader)
        eval_loss, eval_time = 0.0, time.time()

        for _ in range(n_eval_steps):
            if self.args.pipe_parallel_size > 0:
                loss = self.engine.eval_batch(data_iter=eval_iterator)
            else:
                loss = self.eval_batch_without_pipe_parallel(data_iter=eval_iterator)
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
