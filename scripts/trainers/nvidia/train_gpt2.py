# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from transformers import GPT2Config, GPT2LMHeadModel

from archai.trainers.nlp.nvidia_trainer import NvidiaTrainer
from archai.trainers.nlp.nvidia_training_args import NvidiaTrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a GPT-2 using the NVIDIA trainer.")

    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("-nc", "--no_cuda", action="store_true", help="Whether CUDA should not be used.")

    parser.add_argument("-ls", "--logging_steps", type=int, default=10, help="Number of steps between logs.")

    parser.add_argument("-es", "--eval_steps", type=int, default=100, help="Number of steps between evaluations.")

    parser.add_argument("-dn", "--dataset_name", type=str, default="wt103", help="Name of the dataset.")

    parser.add_argument("-vt", "--vocab_type", type=str, default="gpt2", help="Name of the vocabulary/tokenizer.")

    parser.add_argument("-vs", "--vocab_size", type=int, default=10000, help="Size of the vocabulary.")

    parser.add_argument("-bsz", "--global_batch_size", type=int, default=256, help="Global batch size.")

    parser.add_argument("-seq", "--seq_len", type=int, default=192, help="Sequence length.")

    parser.add_argument("-st", "--strategy", type=str, default="ddp", help="Distributed training strategy.")

    parser.add_argument("-n", "--max_steps", type=int, default=250, help="Maximum number of training steps.")

    parser.add_argument(
        "-ng", "--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps."
    )

    parser.add_argument("-o", "--optim", type=str, default="jitlamb", help="Name of the optimizer.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    training_args = NvidiaTrainingArguments(
        "nvidia-gpt2",
        seed=args.seed,
        no_cuda=args.no_cuda,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        dataset_name=args.dataset_name,
        vocab_type=args.vocab_type,
        vocab_size=args.vocab_size,
        global_batch_size=args.global_batch_size,
        seq_len=args.seq_len,
        strategy=args.strategy,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
    )

    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.seq_len,
        n_embd=512,
        n_layer=16,
        n_head=8,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config=config)

    trainer = NvidiaTrainer(model=model, args=training_args)
    trainer.train()
