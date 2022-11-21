# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable trainer with huggingface/accelerate.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers.optimization import get_scheduler
from transformers.data.data_collator import DataCollatorWithPadding, default_data_collator
from transformers.trainer_utils import seed_worker, enable_full_determinism, set_seed
from transformers.training_args import TrainingArguments
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


class HfAccelerateTrainer:
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

        self.accelerator = Accelerator()

        if args is None:
            args = TrainingArguments(output_dir="tmp_trainer")
        self.args = args

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

    def get_dataloader(self, dataset, sampler):
        """"""
        
        return DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
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
        
    def create_scheduler(self, num_training_steps):
        """"""

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def create_optimizer_and_scheduler(self):
        """"""
        
        self.create_optimizer()
        self.create_scheduler()

    def training_step(self):
        """"""
        pass

    def train(self):
        """"""
        pass

    def prediction_step(self):
        """"""
        pass

    def predict(self):
        """"""
        pass

    def evaluate(self):
        """"""
        pass
