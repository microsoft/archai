# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
from overrides import overrides
from torch.utils.data import Dataset

from archai.api.trainer_base import TrainerBase
from archai.trainers.cv.pt_training_args import TorchTrainingArguments


class TorchTrainer(TrainerBase):
    """PyTorch trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: Optional[TorchTrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """Initialize the trainer."""

        super().__init__()

        self.model = model

        if args is None:
            self.args = TorchTrainingArguments()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        #
        self._setup()

    def _setup(self) -> None:
        """Setup the trainer."""

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _train_step(self, inputs, labels) -> None:
        """Training step."""

        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        loss = self.loss_fn(outputs, labels)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    @overrides
    def train(self) -> None:
        total_loss = 0.0

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True)

        self.model.train()
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
            total_loss += self._train_step(inputs, labels)

            if idx % 10 == 0:
                print(f"Batch {idx} loss: {total_loss / (idx + 1)}")

            if idx % self.args.eval_steps == 0:
                eval_loss, eval_acc = self.evaluate()

    def _eval_step(self, inputs, labels) -> None:
        """Evaluation step."""

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)

        return loss.item(), 0.0

    @overrides
    def evaluate(self, eval_dataset=None) -> None:
        eval_dataset = eval_dataset if eval_dataset else self.eval_dataset
        assert eval_dataset is not None, "`eval_dataset` has not been provided."

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)

        eval_loss, eval_acc = 0.0, 0.0

        self.model.eval()
        for idx, (inputs, labels) in enumerate(eval_loader):
            inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
            loss, acc = self._eval_step(inputs, labels)

            eval_loss += loss
            eval_acc += acc

        self.model.train()

        eval_loss /= idx
        eval_acc /= idx

        return eval_loss, eval_acc

    @overrides
    def predict(self) -> None:
        return super().predict()
