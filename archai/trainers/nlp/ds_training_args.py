# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Union

import deepspeed
import torch

from archai.common.file_utils import get_full_path


@dataclass
class DsTrainingArguments:
    """Define arguments used in the DeepSpeed training pipeline.

    Args:
        output_dir: Output folder.
        ds_config: DeepSpeed configuration (dictionary or path to JSON file).
        do_eval: Whether to enable evaluation.
        max_steps: Maximum number of training steps.
        logging_steps: Number of steps between logs.
        save_steps: Number of steps between checkpoints.
        seed: Random seed.
        local_rank: Rank of process.
        backend: Distributed training backend.
        eval_steps: Number of steps between evaluations.
        pipe_parallel: Whether to use pipeline parallelism.
        pipe_parallel_size: Size of pipeline parallelism.
        pipe_parallel_loss_fn: Loss function for pipeline parallelism.
        pipe_parallel_partition_method: Partition method for pipeline parallelism.
        pipe_parallel_activation_checkpoint_steps: Number of steps between pipeline parallelism activation checkpoins.

    """

    output_dir: str = field(metadata={"help": "Output folder."})

    ds_config: Union[dict, str] = field(
        default_factory=dict, metadata={"help": "DeepSpeed configuration (dictionary or path to JSON file)."}
    )

    do_eval: bool = field(default=True, metadata={"help": "Whether to enable evaluation."})

    max_steps: int = field(default=1, metadata={"help": "Maximum number of training steps."})

    logging_steps: int = field(default=10, metadata={"help": "Number of steps between logs."})

    save_steps: int = field(default=500, metadata={"help": "Number of steps between checkpoints."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    local_rank: int = field(default=os.getenv("LOCAL_RANK", -1), metadata={"help": "Rank of process."})

    backend: int = field(default="nccl", metadata={"help": "Distributed training backend."})

    eval_steps: int = field(default=500, metadata={"help": "Number of steps between evaluations."})

    eval_max_steps: int = field(default=None, metadata={"help": "Number of maximum steps during evaluation."})

    pipe_parallel_size: int = field(default=1, metadata={"help": "Size of pipeline parallelism."})

    pipe_parallel_loss_fn: callable = field(default=None, metadata={"help": "Loss function for pipeline parallelism."})

    pipe_parallel_partition_method: str = field(
        default="parameters", metadata={"help": "Partition method for pipeline parallelism."}
    )

    pipe_parallel_activation_checkpoint_steps: int = field(
        default=0, metadata={"help": "Number of steps between pipeline parallelism activation checkpoins."}
    )

    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin the data loader memory."})

    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of subprocesses to use for data loading."})

    def __post_init__(self) -> None:
        """Override post-initialization with custom instructions."""

        self.output_dir = get_full_path(self.output_dir)

        if isinstance(self.ds_config, str):
            with open(self.ds_config, "r") as f:
                self.ds_config = json.load(f)

        torch.manual_seed(self.seed)
        deepspeed.runtime.utils.set_random_seed(self.seed)

        self.local_rank = int(self.local_rank)
        torch.cuda.set_device(self.local_rank)

    def to_dict(self) -> Dict[str, Any]:
        """Convert attributes into a dictionary representation.

        Returns:
            Attributes encoded as a dictionary.

        """

        return asdict(self)
