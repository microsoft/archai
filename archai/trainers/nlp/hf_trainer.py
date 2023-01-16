# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides
from transformers.trainer import Trainer

from archai.api.trainer_base import TrainerBase
from archai.trainers.nlp.hf_training_args import DistillerTrainingArguments


class HfTrainer(Trainer, TrainerBase):
    """Hugging Face trainer."""

    @overrides
    def _rotate_checkpoints(self, use_mtime: Optional[bool] = False, output_dir: Optional[str] = None) -> None:
        """Rotate checkpoints and cache them to Azure Storage.

        The `use_mtime` argument is always set to `False` to avoid having
        multiple checkpoints with the same timestamp when retrieving them
        from Azure Storage. This is because Azure Storage does not support
        sub-second precision for file timestamps.

        Args:
            use_mtime: Whether to use mtime to sort the checkpoints.
            output_dir: Folder to output the checkpoints.

        """

        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Enforces use_mtime=False to avoid identical timestamps
        # when retrieving files from Azure Storage
        use_mtime = False

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True,
        # we could end up deleting the last checkpoint, which
        # we don't do to allow resuming
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            try:
                shutil.rmtree(checkpoint)
            except FileNotFoundError:
                pass


class HfDistillerTrainer(HfTrainer):
    """Hugging Face distillation-based trainer."""

    def __init__(self, teacher_model: torch.nn.Module, **kwargs) -> None:
        """Initializes Hugging Face distillation-based trainer.

        Args:
            teacher_model: Pre-trained teacher model.

        """

        self.teacher_model = teacher_model

        if "args" in kwargs:
            assert isinstance(
                kwargs["args"], DistillerTrainingArguments
            ), "`args` should be an instance of `DistillerTrainingArguments`."
        else:
            kwargs["args"] = DistillerTrainingArguments("tmp")

        super().__init__(**kwargs)

    @overrides
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Override the computation of the loss function.

        The loss is a weighted sum of the student's loss, as computed by
        the original `HfTrainer`, and the KL divergence between the student and
        teacher models.

        Args:
            model: Student model.
            inputs: Input tensors.
            return_outputs: Whether outputs should be returned.

        Returns:
            (loss, outputs) or the loss tensor.

        """

        student_outputs = model(**inputs)

        student_loss = student_outputs["loss"]
        student_logits = student_outputs["logits"]

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs["logits"]

        # Compute the KL divergence and KD losses
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        kl_divergence = kl_loss(
            F.log_softmax(student_logits / self.args.temperature, dim=-1),
            F.softmax(teacher_logits / self.args.temperature, dim=-1),
        )
        kd_loss = self.args.temperature**2 * kl_divergence

        # Weigh the final loss
        loss = self.args.alpha * student_loss + (1 - self.args.alpha) * kd_loss

        return (loss, student_outputs) if return_outputs else loss
