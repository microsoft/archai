# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest
if os.name == "nt":
    pytest.skip(allow_module_level=True)

from archai.common.file_utils import get_full_path
from archai.trainers.nlp.ds_training_args import DsTrainingArguments


def test_ds_training_arguments():
    # Assert that the default values are correct
    args = DsTrainingArguments("logdir")

    assert args.output_dir == get_full_path("logdir")
    assert args.ds_config == {}
    assert args.do_eval == True
    assert args.max_steps == 1
    assert args.logging_steps == 10
    assert args.save_steps == 500
    assert args.seed == 42
    assert args.local_rank == os.getenv("LOCAL_RANK", -1)
    assert args.backend == "nccl"
    assert args.eval_steps == 500
    assert args.eval_max_steps == None
    assert args.pipe_parallel_size == 1
    assert args.pipe_parallel_loss_fn == None
    assert args.pipe_parallel_partition_method == "parameters"
    assert args.pipe_parallel_activation_checkpoint_steps == 0
    assert args.dataloader_pin_memory == True
    assert args.dataloader_num_workers == 0
