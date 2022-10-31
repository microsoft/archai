# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable trainer using NVIDIA-based pipeline.
"""

import itertools
import os
import sys
import time
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from packaging import version
from torch.nn.parallel import DistributedDataParallel

from archai.nlp.datasets.nvidia import distributed_utils
from archai.nlp.datasets.nvidia.corpus import get_lm_corpus
from archai.nlp.datasets.nvidia.lm_iterators import (
    LMMultiFileIterator,
    LMOrderedIterator,
    LMShuffledIterator,
)
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments
from archai.nlp import logging_utils
from archai.nlp.trainers.nvidia.utils.cyclic_cosine_scheduler import CyclicCosineDecayLR
from archai.nlp.trainers.nvidia.utils.optimizers import JITLamb, Lamb

logger = logging_utils.get_logger(__name__)


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

    def _get_dataloader(self, split: str) -> Union[LMOrderedIterator, LMShuffledIterator, LMMultiFileIterator]:
        """Gets a data loader from the pre-loaded dataset.

        Args:
            split: Split of dataset to be retrieved as data loader.

        Returns:
            (Union[LMOrderedIterator, LMShuffledIterator, LMMultiFileIterator]): An instance of
                data loader/iterator based on the loaded dataset.

        """

        return self.dataset.get_iterator(
            split,
            self.args.batch_size,
            self.args.seq_len,
            self.args.device,
        )

    def wrap_model_distributed(self) -> None:
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
        elif scheduler_name == "dev_perf":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.args.decay_rate,
                patience=self.args.patience,
                min_lr=self.args.optimizer_lr_min,
            )
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

    def training_step(self, train_itr, val_itr, last_iter, epoch, last_batch, train_step):
        """"""

        self.model.train()

        train_loss = 0
        labels_tokens = 0
        log_step = 0
        log_start_time = time.time()

        # Changes to make train_iter for lm1b to be properly caught
        if self.args.dataset != "lm1b":
            train_iter = train_itr.get_fixlen_iter(start=last_iter)
        else:
            train_iter = train_itr

        # Supports different autocast signatures and usage of bfloat16
        autocast = torch.autocast(self.args.device.type, enabled=self.args.fp16)
        if version.parse(torch.__version__) >= version.parse("1.10"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast = torch.cuda.amp.autocast(enabled=self.args.fp16, dtype=dtype)

        for batch, (input_ids, labels, _, _) in enumerate(train_iter, start=last_batch + 1):
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
            if self.args.scheduler in ["cosine", "constant", "dev_perf"]:
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
                    f"Batch: {batch} / {train_itr.n_batch} | LR: {lr:.3e} | "
                    f"ms/batch: {avg_elapsed*1000:.1f} | tok/s: {throughput:.0f} | "
                    f"Loss: {loss:.3f}"
                )

            do_periodic_eval = train_step % self.args.eval_interval == 0
            is_final_step = train_step == self.args.max_steps
            interrupted = False  # timeout_handler.interrupted

            if (do_periodic_eval or is_final_step or interrupted) and not self.args.disable_eval:
                eval_start_time = time.time()
                eval_loss = self.evaluation_step(val_itr)
                logger.info(f"Eval loss: {eval_loss}")

                # val_metrix = EvalMetrics(valid_file_stats.word_count, *node_metrix)

                # log_str = "| Eval {:3d} at step {:>8d} | time: {:5.2f}s " \
                #         "| loss {:5.2f} | word ppl {:5.2f}".format(
                #             train_step // self.args.eval_interval,
                #             train_step,
                #             (time.time() - eval_start_time),
                #             val_metrix.avg_loss, val_metrix.word_ppl
                #             )
                last_iter = train_itr.last_iter

                # if self.args.qat:
                #     # Convert the model to a regular FP32 model for saving
                #     model_float = copy.deepcopy(self.model)
                #     model_float = qat_to_float_modules(model_float)
                #     model_to_save = model_float
                #     prefix = "qat_"

                # save_checkpoint(args, model_to_save, model_config, optimizer, scheduler,
                #                 scaler, vocab, epoch, batch, last_iter,
                #                 train_step, best_val_loss, is_best,
                #                 self.args.work_dir, prefix=prefix)

                # dev-performance based learning rate annealing
                # if self.args.scheduler == "dev_perf":
                #     self.scheduler.step(val_metrix.avg_loss)
                #     if self.scheduler_sparse:
                #         self.scheduler_sparse.step(val_metrix.avg_loss)

                # subtract eval time from timers for training
                log_start_time += time.time() - eval_start_time

            if interrupted:
                sys.exit(0)

            if is_final_step:
                break

    def train(self, resume_from_checkpoint=None):
        """"""

        self.create_optimizer()
        self.create_scaler()
        self.create_scheduler()

        self.wrap_model_distributed()

        train_itr = self._get_dataloader("train")
        val_itr = self._get_dataloader("valid")
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

                # self.model.apply(functools.partial(update_dropout, args=args))
                # self.model.apply(functools.partial(update_dropatt, args=args))

                self.wrap_model_distributed()
            except FileNotFoundError:
                pass

        logger.info("Starting training ...")
        logger.debug(f"Training arguments: {self.args.to_dict()}")

        start_time = time.time()
        try:
            for epoch in itertools.count(start=start_epoch):
                if self.args.roll:
                    train_itr.roll(seed=self.args.seed + epoch)
                self.training_step(train_itr, val_itr, last_iter, epoch, last_batch, train_step)
        except KeyboardInterrupt:
            logger.info('Exiting from training ...')

        end_time = time.time()
        logger.info(f'Training time: {((end_time - start_time) / 60):.2f} minutes')

    def evaluation_step(self, eval_iter):
        """"""
        self.model.eval()

        # Evaluation
        total_len, total_loss, total_loss_nomem, steps, total_len_nowarmup, batches = 0, 0.0, 0.0, 0, 0, -1
        start_time = time.time()
        with torch.no_grad():
            mems = None
            for batches, (input_ids, labels, _, _) in enumerate(eval_iter):
                # now without mem
                # loss_nomem = None
                # if eval_nomem:
                numel = input_ids.numel()
                loss = self.dist_model(input_ids, labels=labels).loss
                loss = loss.float().mean()

                total_loss += loss.item()
            total_loss /= batches

                # total_len_nowarmup += numel
                # if warm:
                #     # assert (mems is None) or mems.size(1) == model.mem_len
                #     total_loss += numel * loss.item()
                #     total_len += numel

                #     if eval_nomem:
                #         total_loss_nomem += numel * loss.item()

        elapsed = time.time() - start_time

        self.model.train()

        return total_loss

    def evaluate(self):
        """"""
        n_params = self.model.get_params()
        summary = {"n_all_param": n_params["total"], "n_nonemb_param": n_params["non_embedding"]}

        if not self.args.no_eval and os.path.exists(checkpoint_path):
            # Load the best saved model
            self.model, _, _ = load_model_from_checkpoint(self.args.model_type, checkpoint_path, on_cpu=False)

            # Run on test data
            test_start_time = time.time()
            node_metrix = evaluate(test_itr, self.model, args, eval_nomem=True)
            test_metrix = EvalMetrics(test_file_stats.word_count, *node_metrix)

            test_elapsed = time.time() - test_start_time

            summary.update(
                {
                    "test_word_count": test_metrix.eval_word_count,
                    "test_total_elapsed": test_metrix.total_elapsed,
                    "test_elapsed": test_elapsed,
                    "test_total_loss": test_metrix.total_loss,
                    "test_total_loss_nomem": test_metrix.total_loss_nomem,
                    "test_avg_loss": test_metrix.avg_loss,
                    "test_avg_loss_nomem": test_metrix.avg_loss_nomem,
                    "test_steps": test_metrix.total_steps,
                    "test_len": test_metrix.total_len,
                    "total_len_nowarmup": test_metrix.total_len_nowarmup,
                    "warmup_discount": test_metrix.warmup_discount,
                    "test_word_ppl": test_metrix.word_ppl,
                    "test_word_ppl_nomem": test_metrix.word_ppl_nomem,
                }
            )

            if self.args.dataset in ["enwik8", "text8"]:
                summary["test_bits_per_character"] = test_metrix.bpc
                summary["test_bits_per_character_nomem"] = test_metrix.bpc_nomem
            else:
                summary["test_ppl"] = test_metrix.ppl
                summary["test_ppl_nomem"] = test_metrix.ppl_nomem

        return summary

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
