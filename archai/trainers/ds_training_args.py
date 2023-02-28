# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import deepspeed
import torch


@dataclass
class DsTrainingArguments:
    """Define arguments used in the DeepSpeed training pipeline.

    Args:


    """

    deepspeed_config: str = field(metadata={"help": "Path to the DeepSpeed configuration."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    local_rank: int = field(default=os.getenv("LOCAL_RANK", -1), metadata={"help": "Rank of process."})

    backend: int = field(default="nccl", metadata={"help": "Distributed training backend."})

    max_steps: int = field(default=100, metadata={"help": "Maximum number of training steps."})

    pipeline_parallalelism: bool = field(default=True, metadata={"help": "Whether to use pipeline parallelism."})

    pp_size: int = field(default=1, metadata={"help": "Size of pipeline parallelism."})

    pp_loss_fn: callable = field(default=None, metadata={"help": ""})

    pp_partition_method: str = field(default="parameters", metadata={"help": ""})

    pp_activation_checkpoint_interval: int = field(default=0, metadata={"help": ""})

    def __post_init__(self) -> None:
        """Override post-initialization with custom instructions."""

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
