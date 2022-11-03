# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable trainer using NVIDIA-based pipeline.
"""

import copy
import itertools
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from packaging import version
from torch.nn.parallel import DistributedDataParallel

from archai.nlp import logging_utils
from archai.nlp.datasets.nvidia import distributed_utils
from archai.nlp.datasets.nvidia.corpus import get_lm_corpus
from archai.nlp.quantization.mixed_qat import MixedQAT
from archai.nlp.quantization.qat import prepare_with_qat, qat_to_float_modules
from archai.nlp.trainers.nvidia.cyclic_cosine_scheduler import CyclicCosineDecayLR
from archai.nlp.trainers.nvidia.optimizers import JITLamb, Lamb
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

logger = logging_utils.get_logger(__name__)


def save_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    fp16: bool,
    iterator: int,
    epoch: int,
    batch: int,
    step: int,
    prefix: Optional[str] = None,
    save_all_checkpoints: Optional[bool] = False,
) -> None:
    """Saves a checkpoint that holds enough information to resume the training.

    Args:
        output_dir: Folder where checkpoint should be saved.
        model: Instance of model.
        optimizer: Instance of optimizer.
        scheduler: Instance of scheduler.
        scaler: Instance of scaler.
        fp16: Whether fp16 precision is used or not.
        iterator: Current iterator.
        epoch: Current epoch.
        batch: Current batch.
        step: Current step.
        prefix: Prefix which should be added to the checkpoint's file name.
        save_all_checkpoints: Whether all `eval_interval` steps should be saved or not.

    """

    state = {
        "model_config": model.config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if fp16 else None,
        "iterator": iterator,
        "epoch": epoch,
        "batch": batch,
        "step": step,
    }

    checkpoint_name = prefix + "checkpoint_last.pt"

    with distributed_utils.sync_workers() as rank:
        checkpoint_path = os.path.join(output_dir, checkpoint_name)

        if rank == 0:
            logger.info(f"Saving checkpoint: {checkpoint_path}")
            torch.save(state, checkpoint_path)

            if save_all_checkpoints:
                checkpoint_step_name = prefix + f"checkpoint_{step}.pt"
                checkpoint_step_path = os.path.join(output_dir, checkpoint_step_name)

                logger.info(f"Saving checkpoint: {checkpoint_step_path}")
                shutil.copy(checkpoint_path, checkpoint_step_path)


class NvidiaTrainer:
    """Implements an NVIDIA-based trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: Optional[NvidiaTrainingArguments] = None,
    ) -> None:
        """Initializes by verifying model and training arguments, and loading dataset.

        Args:
            model: Model to be trained or evaluated.
            args: NVIDIA-based training arguments.

        """

        assert isinstance(model, torch.nn.Module), "`model` should be an instance of `torch.nn.Module`."
        self.model = model

        if args is None:
            args = NvidiaTrainingArguments("tmp_trainer")
        assert isinstance(args, NvidiaTrainingArguments), "`args` should be an instance of `NvidiaTrainingArguments`."
        self.args = args

        self.dataset = get_lm_corpus(
            self.args.dataset_dir,
            self.args.dataset_cache_dir,
            self.args.dataset,
            self.args.vocab,
            vocab_size=self.args.vocab_size,
            refresh_cache=self.args.dataset_refresh_cache,
        )

        self.model.to(self.args.device)

    def load_checkpoint(self, checkpoint_file_path: str) -> Tuple[int, int, int, int]:
        """Loads states from checkpoint.

        Args:
            checkpoint_file_path: Path to the checkpoint.

        Returns:
            (Tuple[int, int, int, int]): Current iterator, epoch, batch and step values.

        """

        try:
            checkpoint = torch.load(checkpoint_file_path, map_location=self.args.device)

            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            if self.args.fp16:
                self.scaler.load_state_dict(checkpoint["amp_state"])

            iterator = checkpoint["iterator"]
            start_epoch = checkpoint["epoch"]
            start_batch = checkpoint["batch"]
            step = checkpoint["step"]

            if step >= self.args.max_steps:
                sys.exit(1)

            return iterator, start_epoch, start_batch, step

        except FileNotFoundError:
            return 0, 0, 0, 0

    def get_dataloader(self, split: str) -> Iterator:
        """Gets a data loader from the pre-loaded dataset.

        Args:
            split: Split of dataset to be retrieved as data loader.

        Returns:
            (Iterator): An instance of data loader/iterator based on the loaded dataset.

        """

        return self.dataset.get_iterator(
            split,
            self.args.batch_size,
            self.args.seq_len,
            self.args.device,
        )

    def create_optimizer(self) -> None:
        """Creates an optimizer and attaches model's parameters."""

        optimizer_name = self.args.optimizer.lower()
        if optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.args.optimizer_lr, momentum=self.optimizer_momentum
            )
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.args.optimizer_lr, weight_decay=self.args.optimizer_weight_decay
            )
        elif optimizer_name == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.optimizer_lr)
        elif optimizer_name == "lamb":
            self.optimizer = Lamb(
                self.model.parameters(), lr=self.args.optimizer_lr, weight_decay=self.args.optimizer_weight_decay
            )
        elif optimizer_name == "jitlamb":
            self.optimizer = JITLamb(
                self.model.parameters(), lr=self.args.optimizer_lr, weight_decay=self.args.optimizer_weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer: {self.args.optimizer} is not implemented yet.")

    def create_scaler(self) -> None:
        """Creates an automatic gradient scaler to support FP16 precision."""

        self.scaler = None
        if self.args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

    def create_scheduler(self) -> None:
        """Creates a learning rate scheduler."""

        scheduler_name = self.args.scheduler_qat if self.args.qat else self.args.scheduler
        if scheduler_name == "cosine":
            if self.args.scheduler_max_steps:
                max_steps = self.args.scheduler_max_steps
            else:
                max_steps = self.args.max_steps
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, max_steps - self.args.scheduler_warmup_steps, eta_min=self.args.scheduler_lr_min
            )
        elif scheduler_name == "inv_sqrt":

            def lr_lambda(step: int) -> float:
                if step == 0 and self.args.scheduler_warmup_steps == 0:
                    return 1.0
                else:
                    return (
                        1.0 / (step**0.5)
                        if step > self.args.scheduler_warmup_steps
                        else step / (self.args.scheduler_warmup_steps**1.5)
                    )

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif scheduler_name == "cyclic_cosine":
            init_decay_steps = int((self.args.max_step - self.args.scheduler_warmup_steps) / 2)
            restart_interval = int((self.args.max_step - self.args.scheduler_warmup_steps) / 4)
            self.scheduler = CyclicCosineDecayLR(
                self.optimizer,
                init_decay_steps,
                self.args.scheduler_lr_min,
                restart_interval,
                warmup_epochs=self.args.scheduler_warmup_steps,
                warmup_start_lr=self.args.optimizer_lr * 0.01,
            )
        elif scheduler_name == "constant":
            pass

    def setup_qat(self) -> None:
        """Setups whether Quantization Aware Training (QAT) should be used or not."""

        if self.args.qat:
            self.model = prepare_with_qat(self.model, onnx_compatible=True)

        if self.args.mixed_qat:
            self.model = MixedQAT(self.model)

    def setup_distributed_training(self) -> None:
        """Setups distributed training."""

        self.dist_model = self.model

        if self.args.strategy == "ddp" and torch.distributed.is_initialized():
            self.dist_model = DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=self.args.find_unused_parameters,
            )

        elif self.args.strategy == "dp":
            self.dist_model = nn.DataParallel(self.model, dim=1)

    def training_step_chunk(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor, autocast: torch.autocast
    ) -> float:
        """Performs the training of a single chunk.

        Args:
            input_ids: Chunk of input data.
            labels: Chunk of input labels.
            autocast: An autocast instance that automatically performs
                fp16 or bf16 precision.

        Returns:
            (float): Chunk training loss.

        """

        with autocast:
            loss = self.dist_model(input_ids, labels=input_ids)[0]
            loss = loss.float().mean().type_as(loss) / self.args.gradient_accumulation_steps

        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.float().item()

    def training_step(
        self,
        train_dataloader: Iterator,
        eval_dataloader: Iterator,
        iterator: int,
        epoch: int,
        start_batch: int,
        step: int,
    ) -> None:
        """Performs the training over the supplied data loaders.

        Args:
            train_dataloader: Training data iterator.
            eval_dataloader: Validation data iterator.
            iterator: Current iterator.
            epoch: Current epoch.
            start_batch: At which batch training should start.
            step: Current step.

        """

        self.model.train()

        train_loss, log_step, n_labels_tokens = 0.0, 0, 0
        start_time = time.time()

        # `lm1b` uses a different style of data loader
        if self.args.dataset != "lm1b":
            train_iterator = train_dataloader.get_fixlen_iter(start=iterator)
        else:
            train_iterator = train_dataloader

        # Supports `bf16` based on PyTorch version and CUDA availability
        autocast = torch.autocast(self.args.device.type, enabled=self.args.fp16)
        if version.parse(torch.__version__) >= version.parse("1.10"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast = torch.cuda.amp.autocast(enabled=self.args.fp16, dtype=dtype)

        for batch, (input_ids, labels, _, _) in enumerate(train_iterator, start=start_batch + 1):
            log_step += 1
            n_labels_tokens += labels.numel()

            for param in self.model.parameters():
                param.grad = None

            # Splits into chunks for gradient accumulation
            input_ids_chunks = torch.chunk(input_ids, self.args.gradient_accumulation_steps, 0)
            labels_chunks = torch.chunk(labels, self.args.gradient_accumulation_steps, 0)

            for i in range(self.args.gradient_accumulation_steps):
                input_ids_chunk = input_ids_chunks[i].contiguous()
                labels_chunk = labels_chunks[i].contiguous()

                train_loss_chunk = self.training_step_chunk(
                    input_ids_chunk,
                    labels_chunk,
                    autocast,
                )
                train_loss += train_loss_chunk

            if self.args.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optimizer_clip)

            if self.args.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Learning rate annealing
            step += 1
            if self.args.scheduler in ["cosine", "constant"]:
                if step < self.args.scheduler_warmup_steps:
                    curr_lr = self.args.optimizer_lr * step / self.args.scheduler_warmup_steps
                    self.optimizer.param_groups[0]["lr"] = curr_lr

                else:
                    if self.args.scheduler == "cosine":
                        self.scheduler.step(step - self.args.scheduler_warmup_steps)
            elif self.args.scheduler in ["inv_sqrt", "cyclic_cosine"]:
                self.scheduler.step(step)

            # Evaluation and checkpoint
            if step % self.args.log_interval == 0:
                elapsed_time = time.time() - start_time

                lr = self.optimizer.param_groups[0]["lr"]

                loss = train_loss / log_step
                loss = distributed_utils.all_reduce_item(loss, op="mean")

                batch_time = elapsed_time / log_step
                batch_time = distributed_utils.all_reduce_item(batch_time, op="max")

                throughput = n_labels_tokens / elapsed_time
                throughput = distributed_utils.all_reduce_item(throughput, op="sum")

                train_loss, log_step, n_labels_tokens = 0.0, 0, 0

                logger.info(
                    f"Epoch: {epoch} | Step: {step} | "
                    f"Batch: {batch} / {train_dataloader.n_batch} | LR: {lr:.3e} | "
                    f"ms/batch: {batch_time*1000:.1f} | tok/s: {throughput:.0f} | "
                    f"Loss: {loss:.3f}"
                )

                start_time = time.time()

            do_periodic_eval = step % self.args.eval_interval == 0
            is_final_step = step == self.args.max_steps

            if (do_periodic_eval or is_final_step) and not self.args.disable_eval:
                eval_loss, eval_time = self.evaluation_step(eval_dataloader)
                eval_loss = distributed_utils.all_reduce_item(eval_loss, op="mean")

                logger.info(
                    f"Eval: {step // self.args.eval_interval} | "
                    f"Step: {step} | Time: {eval_time:.2f}s | "
                    f"Loss: {eval_loss:.3f} | PPL: {math.exp(eval_loss):.3f}"
                )

                iterator = train_dataloader.last_iter
                save_model = copy.deepcopy(self.model)
                prefix = ""

                # Model needs to be converted back to FP32 when using QAT
                if self.args.qat:
                    save_model = qat_to_float_modules(save_model)
                    prefix = "qat-"

                # Saves original FP32 model when using MixedQAT
                if self.args.mixed_qat:
                    save_model = save_model.model
                    prefix = "mixed-qat-"

                save_checkpoint(
                    self.args.output_dir,
                    save_model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.args.fp16,
                    iterator,
                    epoch,
                    batch,
                    step,
                    prefix=prefix,
                    save_all_checkpoints=self.args.save_all_checkpoints,
                )

            if is_final_step:
                break

        return step

    def train(self, checkpoint_file_path: Optional[str] = "") -> Dict[str, Any]:
        """Trains a model.

        Args:
            checkpoint_file_path: Path to the checkpoint that will be used
                to resume the training.

        Returns:
            (Dict[str, Any]): Training-related metrics.

        """

        self.create_optimizer()
        self.create_scaler()
        self.create_scheduler()

        if checkpoint_file_path:
            iterator, start_epoch, start_batch, step = self.load_checkpoint(checkpoint_file_path)
        else:
            iterator, start_epoch, start_batch, step = 0, 0, 0, 0

        self.setup_qat()
        self.setup_distributed_training()

        train_dataloader = self.get_dataloader("train")
        eval_dataloader = self.get_dataloader("valid")

        logger.info("Starting training ...")
        logger.debug(f"Training arguments: {self.args.to_dict()}")

        start_time = time.time()
        try:
            for epoch in itertools.count(start=start_epoch):
                if self.args.iterator_shuffle:
                    train_dataloader.roll(seed=self.args.seed + epoch)

                step = self.training_step(train_dataloader, eval_dataloader, iterator, epoch, start_batch, step)

                iterator, start_batch = 0, 0

                if step == self.args.max_steps:
                    logger.info("End of training ...")
                    break

        except KeyboardInterrupt:
            logger.info("Exiting from training ...")
        end_time = time.time()

        train_time = end_time - start_time
        logger.info(f"Training time: {train_time:.3f} seconds")

    def evaluation_step(self, eval_dataloader: Iterator) -> Tuple[float, float]:
        """Performs the evaluation over the supplied data loader.

        Args:
            eval_dataloader: Evaluation-related data loader.

        Returns:
            (Tuple[float, float]): Evaluation loss and time.

        """

        self.model.eval()

        eval_loss, n_tokens = 0.0, 0
        start_time = time.time()
        with torch.no_grad():
            for _, (input_ids, _, _, warm) in enumerate(eval_dataloader):
                loss = self.model(input_ids, labels=input_ids)[0]
                tokens = input_ids.numel()
                if warm:
                    eval_loss += tokens * loss.float().mean().item()
                    n_tokens += tokens
            eval_loss /= n_tokens
        end_time = time.time()

        self.model.train()

        return eval_loss, end_time - start_time

    def evaluate(self, eval_dataloader: Optional[Iterator] = None) -> Dict[str, Any]:
        """Evaluates a model.

        Args:
            eval_dataloader: Evaluation-based data loader. If not supplied, it will
                default to the one available in pre-loaded dataset.

        Returns:
            (Dict[str, Any]): Evaluation-related metrics.

        """

        if not eval_dataloader:
            eval_dataloader = self.get_dataloader("test")

        eval_loss, eval_time = self.evaluation_step(eval_dataloader)

        eval_metrics = {
            "eval_time": eval_time,
            "eval_loss": eval_loss,
            "eval_ppl": math.exp(eval_loss),
            "eval_bpc": eval_loss / math.log(2),
        }

        return eval_metrics
