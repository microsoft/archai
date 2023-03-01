# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Union

import deepspeed
import torch


@dataclass
class DsTrainingArguments:
    """Define arguments used in the DeepSpeed training pipeline.

    Args:


    """

    config: Union[dict, str] = field(metadata={"help": "DeepSpeed configuration (dictionary or path to JSON file)."})

    output_dir: str = field(default="~/logdir", metadata={"help": "Output folder."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    local_rank: int = field(default=os.getenv("LOCAL_RANK", -1), metadata={"help": "Rank of process."})

    backend: int = field(default="nccl", metadata={"help": "Distributed training backend."})

    max_steps: int = field(default=100, metadata={"help": "Maximum number of training steps."})

    do_eval: bool = field(default=True, metadata={"help": "Whether to enable evaluation."})

    eval_steps: int = field(default=100, metadata={"help": "Number of steps between evaluations."})

    save_steps: int = field(default=500, metadata={"help": "Number of steps between checkpoints."})

    pipeline_parallalelism: bool = field(default=True, metadata={"help": "Whether to use pipeline parallelism."})

    pp_size: int = field(default=1, metadata={"help": "Size of pipeline parallelism."})

    pp_loss_fn: callable = field(default=None, metadata={"help": ""})

    pp_partition_method: str = field(default="parameters", metadata={"help": ""})

    pp_activation_checkpoint_interval: int = field(default=0, metadata={"help": ""})

    def __post_init__(self) -> None:
        """Override post-initialization with custom instructions."""

        if isinstance(self.config, str):
            with open(self.config, "r") as f:
                self.config = json.load(f)

        torch.manual_seed(self.seed)
        deepspeed.runtime.utils.set_random_seed(self.seed)

        self.local_rank = int(self.local_rank)
        torch.cuda.set_device(self.local_rank)

        # self.batch_size

    @property
    def batch_size(self) -> int:
        if "train_micro_batch_size_per_gpu" in self.config:
            return self.config.train_micro_batch_size_per_gpu

        # if "train_batch_size" in self.config:

        # print(self.config)
        # raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert attributes into a dictionary representation.

        Returns:
            Attributes encoded as a dictionary.

        """

        return asdict(self)
