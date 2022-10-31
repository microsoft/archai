# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable trainer using NVIDIA-based pipeline.
"""

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

from archai.nlp.datasets.nvidia import distributed_utils
from archai.nlp.datasets.nvidia.corpus import get_lm_corpus
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments
from archai.nlp import logging_utils
from archai.nlp.quantization.mixed_qat import MixedQAT
from archai.nlp.trainers.nvidia.utils.cyclic_cosine_scheduler import CyclicCosineDecayLR
from archai.nlp.trainers.nvidia.utils.optimizers import JITLamb, Lamb

logger = logging_utils.get_logger(__name__)


def save_checkpoint(output_dir, model, optimizer, scheduler, scaler, fp16, epoch, batch, last_iter, train_step, prefix, save_all):
    """
    """

    # FP32 model is saved instead of MixedQAT wrapper
    if isinstance(model, MixedQAT):
        model = model.model

    state = {
        "model_config": model.config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if fp16 else None,
        "epoch": epoch,
        "batch": batch,
        "last_iter": last_iter,
        "train_step": train_step,
        }

    checkpoint_name =  prefix + "checkpoint_last.pt"

    with distributed_utils.sync_workers() as rank:
        checkpoint_path = os.path.join(output_dir, checkpoint_name)

        if rank == 0:
            logger.info(f"Saving checkpoint: {checkpoint_path}")
            torch.save(state, checkpoint_path)

            if save_all:
                checkpoint_step_name = prefix + f"checkpoint_{train_step}.pt"
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
        """Initializes by verifying model, training arguments and loading dataset.

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

    def setup_distributed_training(self) -> None:
        """Wraps a model to support distributed training."""

        self.dist_model = self.model

        if self.args.multi_gpu == "ddp" and torch.distributed.is_initialized():
            self.dist_model = DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=self.args.find_unused_parameters,
            )

        elif self.args.multi_gpu == "dp":
            self.dist_model = nn.DataParallel(self.model, dim=1)

    def create_optimizer(self) -> None:
        """Creates an optimizer and attaches model"s parameters."""

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
            loss = self.dist_model(input_ids, labels=labels).loss
            loss = loss.float().mean().type_as(loss) / self.args.gradient_accumulation_steps

        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.float().item()

    def training_step(self, train_dataloader: Iterator, eval_dataloader: Iterator, last_iter: int, epoch: int, last_batch: int, train_step: int) -> None:
        """"""

        self.model.train()

        train_loss = 0
        labels_tokens = 0
        log_step = 0
        log_start_time = time.time()

        # Changes to make train_dataloader for lm1b to be properly caught
        if self.args.dataset != "lm1b":
            train_iterator = train_dataloader.get_fixlen_iter(start=last_iter)
        else:
            train_iterator = train_dataloader

        # Supports different autocast signatures and usage of bfloat16
        autocast = torch.autocast(self.args.device.type, enabled=self.args.fp16)
        if version.parse(torch.__version__) >= version.parse("1.10"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast = torch.cuda.amp.autocast(enabled=self.args.fp16, dtype=dtype)

        for batch, (input_ids, labels, _, _) in enumerate(train_iterator, start=last_batch + 1):
            log_step += 1
            labels_tokens += labels.numel()

            for param in self.model.parameters():
                param.grad = None

            # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
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

            # step-wise learning rate annealing
            train_step += 1
            if self.args.scheduler in ["cosine", "constant"]:
                # linear warmup stage
                if train_step < self.args.scheduler_warmup_steps:
                    curr_lr = self.args.optimizer_lr * train_step / self.args.scheduler_warmup_steps
                    self.optimizer.param_groups[0]["lr"] = curr_lr
                    
                else:
                    if self.args.scheduler == "cosine":
                        self.scheduler.step(train_step - self.args.scheduler_warmup_steps)
            elif self.args.scheduler in ["inv_sqrt", "cyclic_cosine"]:
                self.scheduler.step(train_step)

            if train_step % self.args.log_interval == 0:
                loss = train_loss / log_step
                loss = distributed_utils.all_reduce_item(loss, op="mean")
                train_loss = 0

                elapsed = time.time() - log_start_time
                avg_elapsed = elapsed / log_step
                avg_elapsed = distributed_utils.all_reduce_item(avg_elapsed, op="max")
                log_start_time = time.time()
                log_step = 0

                lr = self.optimizer.param_groups[0]["lr"]
                throughput = labels_tokens / elapsed
                throughput = distributed_utils.all_reduce_item(throughput, op="sum")
                labels_tokens = 0

                logger.info(
                    f"Epoch: {epoch} | Step: {train_step} | "
                    f"Batch: {batch} / {train_dataloader.n_batch} | LR: {lr:.3e} | "
                    f"ms/batch: {avg_elapsed*1000:.1f} | tok/s: {throughput:.0f} | "
                    f"Loss: {loss:.3f}"
                )

            do_periodic_eval = train_step % self.args.eval_interval == 0
            is_final_step = train_step == self.args.max_steps
            interrupted = False  # timeout_handler.interrupted

            if (do_periodic_eval or is_final_step or interrupted) and not self.args.disable_eval:
                eval_loss, eval_time = self.evaluation_step(eval_dataloader)
                logger.info(
                    f"Eval: {train_step // self.args.eval_interval} | "
                    f"Step: {train_step} | Time: {eval_time:.2f}s | "
                    f"Loss: {eval_loss:.3f} | PPL: {math.exp(eval_loss):.3f}"
                )

                # val_metrix = EvalMetrics(valid_file_stats.word_count, *node_metrix)

                # log_str = "| Eval {:3d} at step {:>8d} | time: {:5.2f}s " \
                #         "| loss {:5.2f} | word ppl {:5.2f}".format(
                #             train_step // self.args.eval_interval,
                #             train_step,
                #             (time.time() - eval_start_time),
                #             val_metrix.avg_loss, val_metrix.word_ppl
                #             )
                last_iter = train_dataloader.last_iter

                # if self.args.qat:
                #     # Convert the model to a regular FP32 model for saving
                #     model_float = copy.deepcopy(self.model)
                #     model_float = qat_to_float_modules(model_float)
                #     model_to_save = model_float
                #     prefix = "qat_"

                save_checkpoint(
                    self.args.output_dir,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.args.fp16,
                    epoch,
                    batch,
                    last_iter,
                    train_step,
                    "",
                    None
                )

            if interrupted:
                sys.exit(0)

            if is_final_step:
                break

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Trains a model.
        
        Args:
            resume_from_checkpoint: Path to the checkpoint that will be used
                to resume the training.

        Returns:
            (Dict[str, Any]): Training-related metrics.
            
        """

        self.create_optimizer()
        self.create_scaler()
        self.create_scheduler()

        train_dataloader = self.get_dataloader("train")
        eval_dataloader = self.get_dataloader("valid")

        last_iter = 0
        last_batch = 0
        train_step = 0
        start_epoch = 1

        if resume_from_checkpoint:
            try:
                checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)

                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                if self.args.fp16:
                    self.scaler.load_state_dict(checkpoint["amp_state"])

                train_step = checkpoint["train_step"]
                start_epoch = checkpoint["epoch"]
                last_batch = checkpoint["batch"]
                last_iter = checkpoint["last_iter"]

                if train_step >= self.args.max_steps:
                    sys.exit(1)

            except FileNotFoundError:
                pass

        self.setup_distributed_training()
        
        logger.info("Starting training ...")
        logger.debug(f"Training arguments: {self.args.to_dict()}")

        start_time = time.time()
        try:
            for epoch in itertools.count(start=start_epoch):
                if self.args.roll:
                    train_dataloader.roll(seed=self.args.seed + epoch)
                self.training_step(train_dataloader, eval_dataloader, last_iter, epoch, last_batch, train_step)
        except KeyboardInterrupt:
            logger.info("Exiting from training ...")
        end_time = time.time()

        logger.info(f"Training time: {((end_time - start_time) / 60):.2f} minutes")

        train_metrics = {

        }

        return train_metrics

    def evaluation_step(self, eval_dataloader: Iterator) -> Tuple[float, float]:
        """Performs an evaluation over the supplied dataloader.
        
        Args:
            eval_dataloader: Evaluation-related dataloader.

        Returns:
            (Tuple[float, float]): Evaluation loss and time.
            
        """

        self.model.eval()

        eval_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            for batches, (input_ids, labels, _, _) in enumerate(eval_dataloader):
                loss = self.model(input_ids, labels=labels).loss
                eval_loss += loss.float().mean().item()
            eval_loss /= batches
        end_time = time.time()

        self.model.train()

        return eval_loss, end_time - start_time

    def evaluate(self, eval_dataloader=None) -> Dict[str, Any]:
        """Evaluates a model.

        Args:
            eval_dataloader: Evaluation-based dataloader. If not supplied, it will
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
            "eval_ppl": math.exp(eval_loss)
        }

        return eval_metrics

    # def post_train_with_qat(self):
    #     """"""
    #     # Creates a dictionary of replacement configs
    #     replace_model_config = {
    #         "dropout": 0.0,
    #         "dropatt": 0.0
    #     }

    #     # Loads the model from the best pre-trained checkpoint
    #     self.model, model_config, _ = load_model_from_checkpoint(self.args.model_type, checkpoint_path, replace_model_config=replace_model_config, on_cpu=False)

    #     # Prepares the model with QAT (also allows for distributed training)
    #     model = prepare_with_qat(self.model, onnx_compatible=True)
    #     model = model.to(self.args.device)
    #     para_self.model, model = distributed_model()

    #     # QAT-based arguments
    #     self.args.restart = None
    #     self.args.qat = True
    #     self.args.max_steps = 10000
    #     self.args.optimizer_lr = self.args.optimizer_lr / 100
    #     self.args.scheduler_lr_min = self.args.scheduler_lr_min / 100
    #     self.args.eval_interval = 1000
    #     self.args.scheduler_warmup_steps = 1000
    #     self.args.optimizer = "adam"

    #     # re-create optimizer
    #     optimizer, optimizer_sparse = create_optimizer()

    #     # re-create scheduler
    #     scheduler, scheduler_sparse = create_scheduler()

    #     # Performs a QAT fine-tuning
    #     training_time, best_val_loss, meters = train_main()
