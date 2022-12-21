# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments


def test_nvidia_training_arguments():
    # Assert that the default values are correct
    args = NvidiaTrainingArguments("tmp", no_cuda=True)
    assert args.experiment_name == "tmp"
    assert args.checkpoint_file_path == ""
    assert os.path.basename(os.path.dirname(args.output_dir)) == "logdir"
    assert args.seed == 42
    assert args.no_cuda is True
    assert args.logging_steps == 10
    assert args.do_eval is True
    assert args.eval_steps == 100
    assert args.save_all_checkpoints is False
    assert args.dataset == "wt103"
    assert os.path.basename(os.path.normpath(args.dataset_dir)) == "wikitext-103"
    assert os.path.basename(os.path.normpath(args.dataset_cache_dir)) == "cache"
    assert args.dataset_refresh_cache is False
    assert args.vocab == "gpt2"
    assert args.vocab_size == 10000
    assert args.iterator_roll is True
    assert args.global_batch_size == 256
    assert args.per_device_global_batch_size is None
    assert args.seq_len == 192
    assert args.strategy == "ddp"
    assert args.local_rank == 0
    assert args.find_unused_parameters is False
    assert args.max_steps == 40000
    assert args.gradient_accumulation_steps == 1
    assert not args.fp16
    assert args.optim == "jitlamb"
    assert args.learning_rate == 0.01
    assert args.weight_decay == 0.0
    assert args.momentum == 0.0
    assert args.max_grad_norm == 0.25
    assert args.lr_scheduler_type == "cosine"
    assert args.lr_qat_scheduler_type == "cosine"
    assert args.lr_scheduler_max_steps is None
    assert args.lr_scheduler_warmup_steps == 1000
    assert args.lr_scheduler_patience == 0
    assert args.lr_scheduler_min_lr == 0.001
    assert args.lr_scheduler_decay_rate == 0.5
    assert not args.qat
    assert not args.mixed_qat

    os.rmdir("tmp")
