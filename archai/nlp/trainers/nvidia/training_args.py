# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable training arguments using NVIDIA-based pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch

from archai.nlp.datasets.nvidia import corpus_utils, distributed_utils


@dataclass
class NvidiaTrainingArguments:
    """Implements a data class that defines arguments used in the NVIDIA training pipeline."""

    experiment_name: str = field(metadata={"help": "Name of the experiment."})

    checkpoint_file_path: str = field(default="", metadata={"help": "Path to the checkpoint file."})

    output_dir: str = field(default="~/logdir", metadata={"help": "Output folder."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    no_cuda: bool = field(default=False, metadata={"help": "Whether CUDA should not be used."})

    logging_steps: int = field(default=10, metadata={"help": "Number of steps between logs."})

    do_eval: bool = field(default=True, metadata={"help": "Whether to enable evaluation."})

    eval_steps: int = field(default=100, metadata={"help": "Number of steps between evaluations."})

    save_all_checkpoints: bool = field(
        default=False, metadata={"help": "Whether to save all checkpoints from `eval_steps` steps."}
    )

    dataset: str = field(default="wt103", metadata={"help": "Name of the dataset."})

    dataset_dir: str = field(default="", metadata={"help": "Dataset folder."})

    dataset_cache_dir: str = field(default="cache", metadata={"help": "Dataset cache folder."})

    dataset_refresh_cache: bool = field(default=False, metadata={"help": "Whether cache should be refreshed."})

    vocab: str = field(default="gpt2", metadata={"help": "Name of the tokenizer."})

    vocab_size: int = field(default=10000, metadata={"help": "Size of the vocabulary"})

    iterator_roll: bool = field(default=True, metadata={"help": "Whether iterator should be rolled."})

    global_batch_size: int = field(default=256, metadata={"help": "Global batch size."})

    per_device_global_batch_size: int = field(default=None, metadata={"help": "Individual GPU batch size."})

    seq_len: int = field(default=192, metadata={"help": "Sequence length."})

    strategy: str = field(default="ddp", metadata={"help": "Multi-GPU strategy."})

    local_rank: int = field(default=os.getenv("LOCAL_RANK", 0), metadata={"help": "Local rank of process."})

    find_unused_parameters: bool = field(default=False, metadata={"help": "Whether unused parameters should be found."})

    max_steps: int = field(default=40000, metadata={"help": "Maximum number of training steps."})

    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of gradient accumulation steps."})

    fp16: bool = field(default=False, metadata={"help": "Whether FP16 precision should be used."})

    optim: str = field(default="jitlamb", metadata={"help": "Name of the optimizer."})

    learning_rate: float = field(default=0.01, metadata={"help": "Optimizer learning rate."})

    weight_decay: float = field(default=0.0, metadata={"help": "Optimizer weight decay."})

    momentum: float = field(default=0.0, metadata={"help": "Optimizer momentum."})

    max_grad_norm: float = field(default=0.25, metadata={"help": "Optimizer gradients clipping value."})

    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Name of the scheduler."})

    lr_qat_scheduler_type: str = field(default="cosine", metadata={"help": "Name of the QAT-based scheduler."})

    lr_scheduler_max_steps: int = field(default=None, metadata={"help": "Maximum number of scheduler steps."})

    lr_scheduler_warmup_steps: int = field(default=1000, metadata={"help": "Number of scheduler warmup steps."})

    lr_scheduler_patience: float = field(default=0, metadata={"help": "Scheduler patience."})

    lr_scheduler_min_lr: float = field(default=0.001, metadata={"help": "Scheduler minimum learning rate."})

    lr_scheduler_decay_rate: float = field(default=0.5, metadata={"help": "Scheduler decay rate."})

    qat: bool = field(default=False, metadata={"help": "Whether QAT should be used during training."})

    mixed_qat: bool = field(default=False, metadata={"help": "Whether MixedQAT should be used during training."})

    @property
    def device(self) -> torch.device:
        """PyTorch device.

        Returns:
            (torch.device): Instance of PyTorch device class.

        """

        return torch.device("cuda" if not self.no_cuda else "cpu")

    def __post_init__(self) -> None:
        """Overrides post-initialization with custom instructions."""

        assert not (self.qat and self.mixed_qat), "`qat` and `mixed_qat` should not be used together."

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.local_rank = int(self.local_rank)
        if not self.no_cuda:
            torch.cuda.set_device(self.local_rank)
            distributed_utils.init_distributed(True)

        (
            self.dataset_dir,
            self.output_dir,
            self.checkpoint_file_path,
            self.dataset_cache_dir,
        ) = corpus_utils.create_dirs(
            self.dataset_dir,
            self.dataset,
            self.experiment_name,
            self.output_dir,
            self.checkpoint_file_path,
            self.dataset_cache_dir,
        )

        with distributed_utils.sync_workers() as rank:
            if rank == 0:
                os.makedirs(self.output_dir, exist_ok=True)

        if self.per_device_global_batch_size is not None:
            world_size = distributed_utils.get_world_size()
            self.global_batch_size = world_size * self.per_device_global_batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Converts attributes into a dictionary representation.

        Returns:
            (Dict[str, Any]): Attributes encoded as a dictionary.

        """

        return self.__dict__
