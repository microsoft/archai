# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from overrides import overrides
from packaging import version
from torch.nn.parallel import DistributedDataParallel

from archai.api.trainer_base import TrainerBase
from archai.common.distributed_utils import all_reduce, sync_workers
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.datasets.nlp.nvidia_data_loader_utils import (
    LMMultiFileIterator,
    LMOrderedIterator,
)
from archai.datasets.nlp.nvidia_dataset_provider import NvidiaDatasetProvider
from archai.quantization.mixed_qat import MixedQAT
from archai.quantization.qat import prepare_with_qat, qat_to_float_modules
from archai.trainers.cyclic_cosine_scheduler import CyclicCosineDecayLR
from archai.trainers.lamb_optimizer import JITLamb, Lamb
from archai.trainers.nlp.nvidia_training_args import NvidiaTrainingArguments

logger = OrderedDictLogger(source=__name__)


def save_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    trainer_state: Dict[str, Any],
    fp16: bool,
    prefix: Optional[str] = "",
    save_all_checkpoints: Optional[bool] = False,
    is_best_model: Optional[bool] = False,
) -> None:
    """Save a checkpoint that holds enough information to resume the training.

    The checkpoint contains the model's configuration and state, the optimizer's state,
    the scheduler's state, the scaler's state (if FP16 precision is used),
    and the trainer's state.

    If `is_best_model` is `True`, the function will also save a copy of the checkpoint
    with the prefix "checkpoint-best".

    If `save_all_checkpoints` is `True`, the function will also save a copy of the checkpoint
    with the step number in the file name.

    Args:
        output_dir: Folder where checkpoint should be saved.
        model: Instance of model.
        optimizer: Instance of optimizer.
        scheduler: Instance of scheduler.
        scaler: Instance of scaler.
        trainer_state: Current trainer state.
        fp16: Whether fp16 precision is used or not.
        prefix: Prefix which should be added to the checkpoint's file name.
        save_all_checkpoints: Whether all `eval_steps` steps should be saved.
        is_best_model: Whether best model should be saved.

    """

    state = {
        "model_config": model.config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if fp16 else None,
        "trainer_state": trainer_state,
    }

    checkpoint_name = prefix + "checkpoint-last.pt"

    with sync_workers() as rank:
        checkpoint_path = os.path.join(output_dir, checkpoint_name)

        if rank == 0:
            logger.info(f"Saving checkpoint: {checkpoint_path}")
            torch.save(state, checkpoint_path)

            if is_best_model:
                checkpoint_step_name = prefix + "checkpoint-best.pt"
                checkpoint_step_path = os.path.join(output_dir, checkpoint_step_name)

                logger.info(f"Saving checkpoint: {checkpoint_step_path}")
                shutil.copy(checkpoint_path, checkpoint_step_path)

            if save_all_checkpoints:
                checkpoint_step_name = prefix + f"checkpoint-{trainer_state['step']}.pt"
                checkpoint_step_path = os.path.join(output_dir, checkpoint_step_name)

                logger.info(f"Saving checkpoint: {checkpoint_step_path}")
                shutil.copy(checkpoint_path, checkpoint_step_path)


class NvidiaTrainer(TrainerBase):
    """NVIDIA-based trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: Optional[NvidiaTrainingArguments] = None,
    ) -> None:
        """Initialize by verifying the model and training arguments, and loading dataset.

        Args:
            model: Model to be trained or evaluated.
            args: NVIDIA-based training arguments. If not provided, a default instance
                of `NvidiaTrainingArguments` will be used.

        """

        assert isinstance(model, torch.nn.Module), "`model` should be an instance of `torch.nn.Module`."
        self.model = model

        if args is None:
            args = NvidiaTrainingArguments("tmp_trainer")
        assert isinstance(args, NvidiaTrainingArguments), "`args` should be an instance of `NvidiaTrainingArguments`."
        self.args = args

        self.dataset_provider = NvidiaDatasetProvider(
            dataset_name=self.args.dataset_name,
            dataset_dir=self.args.dataset_dir,
            cache_dir=self.args.dataset_cache_dir,
            vocab_type=self.args.vocab_type,
            vocab_size=self.args.vocab_size,
            refresh_cache=self.args.dataset_refresh_cache,
        )

        self.model.to(self.args.device)

        self.trainer_state = {
            "iterator": 0,
            "epoch": 0,
            "batch": 0,
            "step": 0,
            "best_eval_loss": 1e300,
            "log_history": [],
        }

    def load_checkpoint(self, checkpoint_file_path: str) -> Tuple[int, int, int, int]:
        """Load states from a checkpoint file.

        Args:
            checkpoint_file_path: Path to the checkpoint file.

        Returns:
            Current iterator, epoch, batch, and step values.

        """

        try:
            checkpoint = torch.load(checkpoint_file_path, map_location=self.args.device)

            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            if self.args.fp16:
                self.scaler.load_state_dict(checkpoint["amp_state"])

            self.trainer_state = checkpoint["trainer_state"]

            iterator = self.trainer_state["iterator"]
            start_epoch = self.trainer_state["epoch"]
            start_batch = self.trainer_state["batch"]
            step = self.trainer_state["step"]

            return iterator, start_epoch, start_batch, step

        except FileNotFoundError:
            return 0, 0, 0, 0

    def _get_dataloader(self, split: str) -> Iterator:
        if split == "train":
            input_ids = self.dataset_provider.get_train_dataset()
        elif split == "valid":
            input_ids = self.dataset_provider.get_val_dataset()
        elif split == "test":
            input_ids = self.dataset_provider.get_test_dataset()
        else:
            raise RuntimeError(f"Split: {split} is not supported yet.")

        if self.args.dataset_name in ["wt2", "wt103"] or self.args.dataset_name.startswith("olx_"):
            return LMOrderedIterator(
                input_ids,
                self.args.global_batch_size,
                self.args.seq_len,
                device=self.args.device,
            )
        elif self.args.dataset_name == "lm1b":
            return LMMultiFileIterator(
                input_ids,
                self.vocab,
                self.args.global_batch_size,
                self.args.seq_len,
                device=self.args.device,
            )
        else:
            raise RuntimeError(f"Dataset: {self.args.dataset_name} is not supported yet.")

    def _create_optimizer(self) -> None:
        optimizer_name = self.args.optim.lower()
        if optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
            )
        elif optimizer_name == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.learning_rate)
        elif optimizer_name == "lamb":
            self.optimizer = Lamb(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
            )
        elif optimizer_name == "jitlamb":
            self.optimizer = JITLamb(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer: {self.args.optim} is not implemented yet.")

    def _create_scaler(self) -> None:
        self.scaler = None
        if self.args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

    def _create_scheduler(self) -> None:
        scheduler_name = self.args.lr_qat_scheduler_type if self.args.qat else self.args.lr_scheduler_type
        if scheduler_name == "cosine":
            if self.args.lr_scheduler_max_steps:
                max_steps = self.args.lr_scheduler_max_steps
            else:
                max_steps = self.args.max_steps
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, max_steps - self.args.lr_scheduler_warmup_steps, eta_min=self.args.lr_scheduler_min_lr
            )
        elif scheduler_name == "inv_sqrt":

            def lr_lambda(step: int) -> float:
                if step == 0 and self.args.lr_scheduler_warmup_steps == 0:
                    return 1.0
                else:
                    return (
                        1.0 / (step**0.5)
                        if step > self.args.lr_scheduler_warmup_steps
                        else step / (self.args.lr_scheduler_warmup_steps**1.5)
                    )

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif scheduler_name == "cyclic_cosine":
            init_decay_steps = int((self.args.max_step - self.args.lr_scheduler_warmup_steps) / 2)
            restart_interval = int((self.args.max_step - self.args.lr_scheduler_warmup_steps) / 4)
            self.scheduler = CyclicCosineDecayLR(
                self.optimizer,
                init_decay_steps,
                self.args.lr_scheduler_min_lr,
                restart_interval,
                warmup_epochs=self.args.lr_scheduler_warmup_steps,
                warmup_start_lr=self.args.learning_rate * 0.01,
            )
        elif scheduler_name == "constant":
            pass

    def _setup_qat(self) -> None:
        if self.args.qat:
            prepare_with_qat(self.model, onnx_compatible=True)

        if self.args.mixed_qat:
            self.model = MixedQAT(self.model)

    def _setup_distributed_training(self) -> None:
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

    def _training_step_chunk(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor, autocast: torch.autocast
    ) -> float:
        with autocast:
            loss = self.dist_model(input_ids, labels=input_ids)[0]
            loss = loss.float().mean().type_as(loss) / self.args.gradient_accumulation_steps

        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.float().item()

    def _training_step(
        self,
        train_dataloader: Iterator,
        eval_dataloader: Iterator,
        iterator: int,
        epoch: int,
        start_batch: int,
        step: int,
    ) -> None:
        self.model.train()

        train_loss, log_step, n_labels_tokens = 0.0, 0, 0
        best_eval_loss = self.trainer_state["best_eval_loss"]

        start_time = time.time()

        # `lm1b` uses a different style of data loader
        if self.args.dataset_name != "lm1b":
            train_iterator = train_dataloader.get_fixlen_iter(start=iterator)
        else:
            train_iterator = train_dataloader

        # Support `bf16` based on PyTorch version and CUDA availability
        autocast = torch.autocast(self.args.device.type, enabled=self.args.fp16)
        if version.parse(torch.__version__) >= version.parse("1.10") and self.args.device.type != "cpu":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast = torch.cuda.amp.autocast(enabled=self.args.fp16, dtype=dtype)

        for batch, (input_ids, labels, _, _) in enumerate(train_iterator, start=start_batch + 1):
            log_step += 1
            n_labels_tokens += labels.numel()

            for param in self.model.parameters():
                param.grad = None

            # Split into chunks for gradient accumulation
            input_ids_chunks = torch.chunk(input_ids, self.args.gradient_accumulation_steps, 0)
            labels_chunks = torch.chunk(labels, self.args.gradient_accumulation_steps, 0)

            for i in range(self.args.gradient_accumulation_steps):
                input_ids_chunk = input_ids_chunks[i].contiguous()
                labels_chunk = labels_chunks[i].contiguous()

                train_loss_chunk = self._training_step_chunk(
                    input_ids_chunk,
                    labels_chunk,
                    autocast,
                )
                train_loss += train_loss_chunk

            if self.args.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            if self.args.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Learning rate annealing
            step += 1
            if self.args.lr_scheduler_type in ["cosine", "constant"]:
                if step < self.args.lr_scheduler_warmup_steps:
                    curr_lr = self.args.learning_rate * step / self.args.lr_scheduler_warmup_steps
                    self.optimizer.param_groups[0]["lr"] = curr_lr

                else:
                    if self.args.lr_scheduler_type == "cosine":
                        self.scheduler.step(step - self.args.lr_scheduler_warmup_steps)
            elif self.args.lr_scheduler_type in ["inv_sqrt", "cyclic_cosine"]:
                self.scheduler.step(step)

            # Logging
            if step % self.args.logging_steps == 0:
                elapsed_time = time.time() - start_time

                lr = self.optimizer.param_groups[0]["lr"]

                loss = train_loss / log_step
                loss = all_reduce(loss, op="mean")

                batch_time = elapsed_time / log_step
                batch_time = all_reduce(batch_time, op="max")

                throughput = n_labels_tokens / elapsed_time
                throughput = all_reduce(throughput, op="sum")

                train_loss, log_step, n_labels_tokens = 0.0, 0, 0

                self.trainer_state["log_history"].append(
                    {
                        "epoch": epoch,
                        "learning_rate": lr,
                        "loss": loss,
                        "ppl": math.exp(loss),
                        "step": step,
                    }
                )

                logger.info(
                    f"Epoch: {epoch} | Step: {step} | "
                    f"Batch: {batch} / {train_dataloader.n_batch} | LR: {lr:.3e} | "
                    f"ms/batch: {batch_time*1000:.1f} | tok/s: {throughput:.0f} | "
                    f"Loss: {loss:.3f} | PPL: {math.exp(loss):.3f}"
                )

                start_time = time.time()

            do_periodic_eval = step % self.args.eval_steps == 0
            is_final_step = step == self.args.max_steps

            # Evaluation and checkpoint
            if (do_periodic_eval or is_final_step) and self.args.do_eval:
                eval_loss, eval_time = self._evaluation_step(eval_dataloader)
                eval_loss = all_reduce(eval_loss, op="mean")

                self.trainer_state["log_history"].append(
                    {
                        "epoch": epoch,
                        "eval_idx": (step // self.args.eval_steps) - 1,
                        "eval_runtime": eval_time,
                        "eval_loss": eval_loss,
                        "eval_ppl": math.exp(eval_loss),
                        "step": step,
                    }
                )

                logger.info(
                    f"Eval: {(step // self.args.eval_steps) - 1} | "
                    f"Step: {step} | Time: {eval_time:.2f}s | "
                    f"Loss: {eval_loss:.3f} | PPL: {math.exp(eval_loss):.3f}"
                )

                iterator = train_dataloader.last_iter
                save_model = copy.deepcopy(self.model)
                prefix = ""

                self.trainer_state["iterator"] = iterator
                self.trainer_state["epoch"] = epoch
                self.trainer_state["batch"] = batch
                self.trainer_state["step"] = step

                # Model needs to be converted back to FP32 when using QAT
                if self.args.qat:
                    qat_to_float_modules(save_model)
                    prefix = "qat-"

                # Save original FP32 model when using MixedQAT
                if self.args.mixed_qat:
                    save_model = save_model.model
                    prefix = "mixed-qat-"

                # Check if current model is the best one
                is_best_model = eval_loss < best_eval_loss
                if is_best_model:
                    best_eval_loss = eval_loss
                    self.trainer_state["best_eval_loss"] = best_eval_loss

                save_checkpoint(
                    self.args.output_dir,
                    save_model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.trainer_state,
                    self.args.fp16,
                    prefix=prefix,
                    save_all_checkpoints=self.args.save_all_checkpoints,
                    is_best_model=is_best_model,
                )

            if is_final_step:
                break

        return step

    @overrides
    def train(self, checkpoint_file_path: Optional[str] = "") -> Dict[str, Any]:
        """Train a model.

        Args:
            checkpoint_file_path: Path to the checkpoint that will be used
                to resume the training.

        Returns:
            Training-related metrics.

        """

        self._create_optimizer()
        self._create_scaler()
        self._create_scheduler()

        if checkpoint_file_path:
            iterator, start_epoch, start_batch, step = self.load_checkpoint(checkpoint_file_path)
        else:
            iterator, start_epoch, start_batch, step = 0, 0, 0, 0

        if step >= self.args.max_steps:
            sys.exit(1)

        self._setup_qat()
        self._setup_distributed_training()

        train_dataloader = self._get_dataloader("train")
        eval_dataloader = self._get_dataloader("valid")

        logger.info("Starting training ...")
        logger.debug(f"Training arguments: {self.args.to_dict()}")

        start_time = time.time()
        try:
            for epoch in itertools.count(start=start_epoch):
                if self.args.iterator_roll:
                    train_dataloader.roll(seed=self.args.seed + epoch)

                step = self._training_step(train_dataloader, eval_dataloader, iterator, epoch, start_batch, step)

                iterator, start_batch = 0, 0

                if step == self.args.max_steps:
                    logger.info("End of training ...")
                    break

        except KeyboardInterrupt:
            logger.info("Exiting from training ...")
        end_time = time.time()

        train_time = end_time - start_time
        logger.info(f"Training time: {train_time:.3f} seconds")

    def _evaluation_step(self, eval_dataloader: Iterator) -> Tuple[float, float]:
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

    @overrides
    def evaluate(self, eval_dataloader: Optional[Iterator] = None) -> Dict[str, Any]:
        """Evaluate a model.

        Args:
            eval_dataloader: Evaluation-based data loader. If not supplied, it will
                default to the one available in pre-loaded dataset.

        Returns:
            Evaluation-related metrics.

        """

        if not eval_dataloader:
            eval_dataloader = self._get_dataloader("test")

        eval_loss, eval_time = self._evaluation_step(eval_dataloader)

        eval_metrics = {
            "eval_time": eval_time,
            "eval_loss": eval_loss,
            "eval_ppl": math.exp(eval_loss),
            "eval_bpc": eval_loss / math.log(2),
        }

        return eval_metrics

    @overrides
    def predict(self) -> None:
        """Predict with a model."""

        pass

    def fine_tune_qat(self, model: Optional[torch.nn.Module] = None, checkpoint_file_path: Optional[str] = "") -> None:
        """Fine-tune a model with QAT.

        Users are allowed to pass in a different model (e.g., without dropout) than the one
        instantiated with `NvidiaTrainer`, as well as a pre-trained checkpoint file to load
        the weights from a previous training.

        Args:
            model: Model to be fine-tuned.
            checkpoint_file_path: Path to the checkpoint used to resume training.

        """

        if model:
            assert isinstance(model, torch.nn.Module), "`model` should be an instance of `torch.nn.Module`."
            self.model = model.to(self.args.device)

        # QAT-based arguments
        self.args.max_steps = 10000
        self.args.eval_steps = 1000
        self.args.optim = "adam"
        self.args.learning_rate /= 100
        self.args.lr_scheduler_min_lr /= 100
        self.args.lr_scheduler_warmup_steps = 1000
        self.args.qat = True
        self.args.mixed_qat = False

        # Re-load the checkpoint and perform the fine-tuning
        self.load_checkpoint(checkpoint_file_path)
        self.train()
