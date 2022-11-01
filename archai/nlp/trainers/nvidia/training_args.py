# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable training arguments using NVIDIA-based pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch

from archai.nlp.datasets.nvidia import distributed_utils, exp_utils


@dataclass
class NvidiaTrainingArguments:
    """Implements a data class that defines arguments used in the NVIDIA training pipeline."""

    experiment_name: str = field(metadata={"help": "Name of the experiment."})

    checkpoint_path: str = field(default="", metadata={"help": ""})

    output_dir: str = field(default="~/logdir", metadata={"help": ""})

    seed: int = field(default=42, metadata={"help": ""})

    use_cuda: bool = field(default=True, metadata={"help": ""})

    log_interval: int = field(default=10, metadata={"help": ""})

    disable_eval: bool = field(default=False, metadata={"help": ""})

    eval_interval: int = field(default=100, metadata={"help": ""})

    save_all_checkpoints: bool = field(default=True, metadata={"help": ""})

    dataset: str = field(default="wt103", metadata={"help": ""})

    dataset_dir: str = field(default="", metadata={"help": ""})

    dataset_cache_dir: str = field(default="cache", metadata={"help": ""})

    dataset_refresh_cache: bool = field(default=False, metadata={"help": ""})

    vocab: str = field(default="gpt2", metadata={"help": ""})

    vocab_size: int = field(default=10000, metadata={"help": ""})

    iterator_shuffle: bool = field(default=False, metadata={"help": ""})

    batch_size: int = field(default=256, metadata={"help": ""})

    local_batch_size: int = field(default=None, metadata={"help": ""})

    seq_len: int = field(default=192, metadata={"help": ""})

    strategy: str = field(default="ddp", metadata={"help": ""})

    local_rank: int = field(default=os.getenv("LOCAL_RANK", 0), metadata={"help": ""})

    find_unused_parameters: bool = field(default=False, metadata={"help": ""})

    max_steps: int = field(default=40000, metadata={"help": ""})

    gradient_accumulation_steps: int = field(default=1, metadata={"help": ""})

    fp16: bool = field(default=False, metadata={"help": ""})

    optimizer: str = field(default="jitlamb", metadata={"help": ""})

    optimizer_lr: float = field(default=0.01, metadata={"help": ""})

    optimizer_weight_decay: float = field(default=0.0, metadata={"help": ""})

    optimizer_momentum: float = field(default=0.0, metadata={"help": ""})

    optimizer_clip: float = field(default=0.25, metadata={"help": ""})

    scheduler: str = field(default="cosine", metadata={"help": ""})

    scheduler_qat: str = field(default="cosine", metadata={"help": ""})

    scheduler_max_steps: int = field(default=None, metadata={"help": ""})

    scheduler_warmup_steps: int = field(default=1000, metadata={"help": ""})

    scheduler_patience: float = field(default=0, metadata={"help": ""})

    scheduler_lr_min: float = field(default=0.001, metadata={"help": ""})

    scheduler_decay_rate: float = field(default=0.5, metadata={"help": ""})

    qat: bool = field(default=False, metadata={"help": ""})

    mixed_qat: bool = field(default=False, metadata={"help": ""})

    @property
    def device(self) -> torch.device:
        """PyTorch device.
        
        Returns:
            (torch.device): Instance of PyTorch device class.
        
        """

        return torch.device("cuda" if self.use_cuda else "cpu")

    def __post_init__(self) -> None:
        """Overrides post-initialization with custom instructions."""

        assert not(self.qat and self.mixed_qat), "`qat` and `mixed_qat` should not be used together."

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.local_rank = int(self.local_rank)
        if self.use_cuda:
            torch.cuda.set_device(self.local_rank)
            distributed_utils.init_distributed(self.use_cuda)

        (
            self.dataset_dir,
            self.output_dir,
            self.checkpoint_path,
            self.dataset_cache_dir,
            _,
        ) = exp_utils.get_create_dirs(
            self.dataset_dir,
            self.dataset,
            self.experiment_name,
            self.output_dir,
            self.checkpoint_path,
            self.dataset_cache_dir,
        )

        with distributed_utils.sync_workers() as rank:
            if rank == 0:
                exp_utils.create_exp_dir(self.output_dir)

        if self.local_batch_size is not None:
            world_size = distributed_utils.get_world_size()
            self.batch_size = world_size * self.local_batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Converts attributes into a dictionary representation.
        
        Returns:
            (Dict[str, Any]): Attributes encoded as a dictionary.
        
        """

        return self.__dict__
