# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.configuration_mem_transformer import (
    MemTransformerConfig,
)
from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.modeling_mem_transformer import (
    MemTransformerLMHeadModel,
)
from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a Memory Transformer using the NVIDIA trainer.")

    parser.add_argument("-s", "--seed", type=int, default=37, help="Random seed.")

    parser.add_argument("-nc", "--no_cuda", action="store_true", help="Whether CUDA should not be used.")

    parser.add_argument("-ls", "--logging_steps", type=int, default=50, help="Number of steps between logs.")

    parser.add_argument("-es", "--eval_steps", type=int, default=1000, help="Number of steps between evaluations.")

    parser.add_argument("-d", "--dataset", type=str, default="wt103", help="Name of the dataset.")

    parser.add_argument("-v", "--vocab", type=str, default="word", help="Name of the vocabulary/tokenizer.")

    parser.add_argument("-vs", "--vocab_size", type=int, default=267735, help="Size of the vocabulary.")

    parser.add_argument("-bsz", "--global_batch_size", type=int, default=256, help="Global batch size.")

    parser.add_argument("-seq", "--seq_len", type=int, default=192, help="Sequence length.")

    parser.add_argument("-st", "--strategy", type=str, default="ddp", help="Distributed training strategy.")

    parser.add_argument("-n", "--max_steps", type=int, default=1000, help="Maximum number of training steps.")

    parser.add_argument(
        "-ng", "--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps."
    )

    parser.add_argument("-o", "--optim", type=str, default="adam", help="Name of the optimizer.")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate.")

    parser.add_argument("-lsw", "--lr_scheduler_warmup_steps", type=int, default=50, help="Scheduler warmup steps.")

    parser.add_argument(
        "-lsm", "--lr_scheduler_min_lr", type=float, default=1e-5, help="Scheduler minimum learning rate."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    training_args = NvidiaTrainingArguments(
        "nvidia-mem-transformer",
        seed=args.seed,
        no_cuda=args.no_cuda,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        dataset=args.dataset,
        vocab=args.vocab,
        vocab_size=args.vocab_size,
        global_batch_size=args.global_batch_size,
        seq_len=args.seq_len,
        strategy=args.strategy,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_warmup_steps=args.lr_scheduler_warmup_steps,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        find_unused_parameters=True,
    )

    config = MemTransformerConfig(
        vocab_size=args.vocab_size,
        n_positions=args.seq_len,
        d_model=256,
        d_embd=256,
        n_head=8,
        d_head=32,
        d_inner=1024,
        div_val=4,
        n_layer=6,
        mem_len=0,
        dropout=0.01,
        dropatt=0.01,
        use_cache=False,
        primer_square=True,
    )
    model = MemTransformerLMHeadModel(config=config)

    trainer = NvidiaTrainer(model=model, args=training_args)
    trainer.train()
