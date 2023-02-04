# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from transformers import OPTConfig, OPTForCausalLM

from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains an OPT using the NVIDIA trainer.")

    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("-nc", "--no_cuda", action="store_true", help="Whether CUDA should not be used.")

    parser.add_argument("-ls", "--logging_steps", type=int, default=10, help="Number of steps between logs.")

    parser.add_argument("-es", "--eval_steps", type=int, default=100, help="Number of steps between evaluations.")

    parser.add_argument("-d", "--dataset", type=str, default="wt103", help="Name of the dataset.")

    parser.add_argument("-v", "--vocab", type=str, default="gpt2", help="Name of the vocabulary/tokenizer.")

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
        "nvidia-opt",
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
    )

    config = OPTConfig(
        vocab_size=args.vocab_size,
        hidden_size=512,
        num_hidden_layers=16,
        ffn_dim=2048,
        max_position_embeddings=args.seq_len,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=8,
        use_cache=False,
    )
    model = OPTForCausalLM(config=config)

    trainer = NvidiaTrainer(model=model, args=training_args)
    trainer.train()
