# Copyright (c) Microsoft Corporation.
# Licensed under the MIT licen

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from archai.common.distributed_utils import (
    get_world_size,
    init_distributed,
    sync_workers,
)
from archai.common.file_utils import get_full_path


def get_amlt_dirs() -> Tuple[str, str]:
    """Get AMLT folders.

    Returns:
        Tuple containing data and output folders.

    """

    data_dir = os.environ.get("AMLT_DATA_DIR", "")
    output_dir = os.environ.get("AMLT_OUTPUT_DIR", "")

    return data_dir, output_dir


def get_default_dataroot() -> str:
    """Get default data root folder.

    Returns:
        Default data root folder.

    """

    is_amlt_available = os.environ.get("AMLT_OUTPUT_DIR", None)

    return "/var/tmp/dataroot" if is_amlt_available else "~/dataroot"


def create_dirs(
    dataroot: str,
    dataset_name: str,
    experiment_name: Optional[str] = "tmp",
    output_dir: Optional[str] = "~/logdir",
    pretrained_path: Optional[str] = "",
    cache_dir: Optional[str] = "",
) -> Tuple[str, str, str, str]:
    """Create dataset-related folders with proper paths.

    Args:
        dataroot: Dataset folder.
        dataset_name: Name of the dataset.
        experiment_name: Experiment name.
        output_dir: Output folder.
        pretrained_path: Path to the pre-trained checkpoint file.
        cache_dir: Dataset cache folder.

    Returns:
        Tuple containing dataset, output, pre-trained checkpoint and cache folders.

    """

    def _get_dataset_dir_name(dataset_name: str) -> str:
        if dataset_name == "wt2":
            return "wikitext-2"
        if dataset_name == "wt103":
            return "wikitext-103"
        if dataset_name == "lm1b":
            return "one-billion-words"
        if dataset_name.startswith("olx_"):
            return dataset_name

        raise RuntimeError(f"Dataset: {dataset_name} is not supported yet.")

    pt_data_dir, pt_output_dir = get_amlt_dirs()
    if pt_output_dir:
        pt_output_dir = os.path.join(pt_output_dir, experiment_name)

    dataroot = dataroot or pt_data_dir or get_default_dataroot()
    dataroot = get_full_path(dataroot)

    dataset_dir = get_full_path(os.path.join(dataroot, "textpred", _get_dataset_dir_name(dataset_name)))
    output_dir = get_full_path(pt_output_dir or os.path.join(output_dir, experiment_name), create_folder=True)

    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(dataset_dir, cache_dir)
    cache_dir = get_full_path(cache_dir, create_folder=True)

    if not os.path.isabs(pretrained_path) and pretrained_path:
        pretrained_path = os.path.join(os.path.dirname(output_dir), pretrained_path)

    return dataset_dir, output_dir, pretrained_path, cache_dir


@dataclass
class NvidiaTrainingArguments:
    """Define arguments used in the NVIDIA training pipeline.

    Args:
        experiment_name: Name of the experiment.
        checkpoint_file_path: Path to the checkpoint file.
        output_dir: Output folder.
        seed: Random seed.
        no_cuda: Whether CUDA should not be used.
        logging_steps: Number of steps between logs.
        do_eval: Whether to enable evaluation.
        eval_steps: Number of steps between evaluations.
        save_all_checkpoints: Whether to save all checkpoints from `eval_steps` steps.
        dataset_name: Name of the dataset.
        dataset_dir: Dataset folder.
        dataset_cache_dir: Dataset cache folder.
        dataset_refresh_cache: Whether cache should be refreshed.
        vocab_type: Name of the vocabulary/tokenizer.
        vocab_size: Size of the vocabulary.
        iterator_roll: Whether iterator should be rolled.
        global_batch_size: Global batch size.
        per_device_global_batch_size: Individual GPU batch size.
        seq_len: Sequence length.
        strategy: Distributed training strategy.
        local_rank: Local rank of process.
        find_unused_parameters: Whether unused parameters should be found.
        max_steps: Maximum number of training steps.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        fp16: Whether FP16 precision should be used.
        optim: Name of the optimizer.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        momentum: Optimizer momentum.
        max_grad_norm: Optimizer gradients clipping value.
        lr_scheduler_type: Name of the scheduler.
        lr_qat_scheduler_type: Name of the QAT-based scheduler.
        lr_scheduler_max_steps: Maximum number of scheduler steps.
        lr_scheduler_warmup_steps: Number of warmup steps for the scheduler.
        lr_scheduler_patience: Scheduler patience.
        lr_scheduler_min_lr: Scheduler minimum learning rate.
        lr_scheduler_decay_rate: Scheduler decay rate.
        qat: Whether QAT should be used during training.
        mixed_qat: Whether MixedQAT should be used during training.

    """

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

    dataset_name: str = field(default="wt103", metadata={"help": "Name of the dataset."})

    dataset_dir: str = field(default="", metadata={"help": "Dataset folder."})

    dataset_cache_dir: str = field(default="cache", metadata={"help": "Dataset cache folder."})

    dataset_refresh_cache: bool = field(default=False, metadata={"help": "Whether cache should be refreshed."})

    vocab_type: str = field(default="gpt2", metadata={"help": "Type of the tokenizer."})

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
        """Return a PyTorch device instance."""

        return torch.device("cuda" if not self.no_cuda else "cpu")

    def __post_init__(self) -> None:
        """Override post-initialization with custom instructions.

        Ensure that `qat` and `mixed_qat` are not used together, set the random seed,
        initialize distributed training, create necessary directories,
        and set the global batch size.

        """

        assert not (self.qat and self.mixed_qat), "`qat` and `mixed_qat` should not be used together."

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.local_rank = int(self.local_rank)
        if not self.no_cuda:
            torch.cuda.set_device(self.local_rank)
            init_distributed(True)

        (self.dataset_dir, self.output_dir, self.checkpoint_file_path, self.dataset_cache_dir,) = create_dirs(
            self.dataset_dir,
            self.dataset_name,
            self.experiment_name,
            self.output_dir,
            self.checkpoint_file_path,
            self.dataset_cache_dir,
        )

        with sync_workers() as rank:
            if rank == 0:
                os.makedirs(self.output_dir, exist_ok=True)

        if self.per_device_global_batch_size is not None:
            world_size = get_world_size()
            self.global_batch_size = world_size * self.per_device_global_batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert attributes into a dictionary representation.

        Returns:
            Attributes encoded as a dictionary.

        """

        return self.__dict__
