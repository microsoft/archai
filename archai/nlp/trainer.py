# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable trainers from huggingface/transformers.
"""

import shutil
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.deepspeed import dep_version_check
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.integrations import is_fairscale_available
from transformers.optimization import Adafactor
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption

from archai.nlp.models.model_utils.optimizers import JITLamb, Lamb

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


class ArchaiTrainer(Trainer):
    """Inherits from Trainer and allows to be customized."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        # Tries to pop the optimizer before overriding
        optimizer = kwargs.pop("optimizer", None)

        super().__init__(*args, **kwargs)

        # Attaches a valid name of the optimizer to be loaded
        self.args.optimizer = optimizer or "adamw"

    def create_optimizer(self) -> None:
        """Overrides the creation of a new optimizer to add extra ones."""

        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
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

            available_optimizers = {
                "adafactor": Adafactor,
                "adamw": AdamW,
                "lamb": Lamb,
                "jitlamb": JITLamb,
            }
            optimizer_cls = available_optimizers[self.args.optimizer]

            if self.args.optimizer == "adafactor":
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            elif self.args.optimizer in ["adamw", "lamb", "jitlamb"]:
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }

            optimizer_kwargs["lr"] = self.args.learning_rate

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(params=optimizer_grouped_parameters, optim=optimizer_cls, **optimizer_kwargs)
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

    def _rotate_checkpoints(self, use_mtime: Optional[bool] = False, output_dir: Optional[str] = None) -> None:
        """Overrides the rotation of checkpoints to allow caching to Azure Storage.

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

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
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
            shutil.rmtree(checkpoint)


class ArchaiDistillerTrainer(ArchaiTrainer):
    """Inherits from ArchaiTrainer and allows a distillation-based training."""

    def __init__(self, teacher_model: torch.nn.Module, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments.

        Args:
            teacher_model: Pre-trained teacher model.

        """

        super().__init__(*args, **kwargs)

        # Knowledge distillation teacher
        self.teacher_model = teacher_model

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Overrides the computation of the loss function.

        Args:
            model: Model subclassed from torch's Module.
            inputs: Dictionary holding the input tensors.
            return_outputs: Whether outputs should be returned.

        Returns:
            (Tuple[torch.Tensor, ...]): Contains (loss, outputs) or just the loss tensor.

        """

        # Gathers the outputs from student
        student_outputs = model(**inputs)

        # Extracts loss and logits from the student
        student_loss = student_outputs["loss"]
        student_logits = student_outputs["prediction_scores"]

        # Inhibits the gradient from working
        with torch.no_grad():
            # Gathers the teacher's outputs and logits
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs["prediction_scores"]

        # Computes the KL divergence loss
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        kl_divergence = kl_loss(
            F.log_softmax(student_logits / self.args.temperature, dim=-1),
            F.softmax(teacher_logits / self.args.temperature, dim=-1),
        )

        # Computes the KD loss
        kd_loss = self.args.temperature**2 * kl_divergence

        # Weighs the final loss
        loss = self.args.alpha * student_loss + (1 - self.args.alpha) * kd_loss

        return (loss, student_outputs) if return_outputs else loss
