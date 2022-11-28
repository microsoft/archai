# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable trainer with huggingface/accelerate.
"""

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.data.data_collator import (
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.optimization import get_scheduler
from transformers.training_args import TrainingArguments

from archai.nlp import logging_utils
from archai.nlp.trainers.hf.accelerate_trainer_utils import (
    ALL_LAYERNORM_LAYERS,
    enable_full_determinism,
    get_parameter_names,
    set_seed,
)

logger = logging_utils.get_logger(__name__)


class AccelerateTrainer:
    """Implements an Accelerate-based trainer."""

    def __init__(
        self,
        model,
        args=None,
        data_collator=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
    ) -> None:
        """"""

        if args is None:
            args = TrainingArguments(output_dir="tmp_trainer")
        self.args = args
        self.args.optim = "adamw"

        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps)

        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        assert isinstance(model, torch.nn.Module), ""

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.model = model

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

    def get_dataloader(self, dataset, sampler=None):
        """"""

        def _seed_worker():
            worker_seed = torch.initial_seed() % 2**32
            set_seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )

    def get_optimizer_cls_and_kwargs(self):
        """"""

        optimizer_kwargs = {"lr": self.args.learning_rate}
        adam_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }

        if self.args.optim == "adamw":
            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)

        return optimizer_cls, optimizer_kwargs

    def create_optimizer(self):
        """"""

        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs()
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_scheduler(self):
        """"""

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(self.args.max_steps),
                num_training_steps=self.args.max_steps,
            )

    def create_optimizer_and_scheduler(self):
        """"""

        self.create_optimizer()
        self.create_scheduler()

    def training_step(self, batch):
        """"""

        outputs = self.model(**batch)

        loss = outputs[0]
        # if self.args.n_gpu > 1:
        # loss = loss.mean()

        # if self.args.gradient_accumulation_steps > 1:
        #     loss /= self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss

    def train(self):
        """"""

        train_dataloder = self.get_dataloader(self.train_dataset)
        self.create_optimizer_and_scheduler()

        self.model, self.optimizer, train_dataloder, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloder, self.lr_scheduler
        )

        self.model.train()

        logger.info("Start training ...")

        total_loss = 0.0

        for step, batch in enumerate(train_dataloder):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()

                total_loss += self.training_step(batch)

                self.optimizer.step()
                self.lr_scheduler.step()

            if step % self.args.logging_steps == 0:
                total_loss /= self.args.logging_steps
                logger.info(f"Step {step} | Loss: {total_loss}")
                total_loss = 0.0

        logger.info("End of training.")

    def prediction_step(self):
        """"""
        pass

    def predict(self):
        """"""
        pass

    def evaluate(self):
        """"""
        pass
