# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    OPTConfig,
    OPTForCausalLM,
    TrainingArguments,
)

from archai.nlp.datasets.hf.loaders import encode_dataset, load_dataset
from archai.nlp.trainers.hf.trainer import HfTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains an OPT using the Huggingface trainer.")

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-dcn",
        "--dataset_config_name",
        type=str,
        default="wikitext-103-raw-v1",
        help="Configuration name of the dataset to use (via the datasets library).",
    )

    parser.add_argument("-seq", "--seq_len", type=int, default=192, help="Sequence length.")

    parser.add_argument("-ls", "--logging_steps", type=int, default=10, help="Number of steps between logs.")

    parser.add_argument("-es", "--eval_steps", type=int, default=100, help="Number of steps between evaluations.")

    parser.add_argument("-bsz", "--per_device_train_batch_size", type=int, default=64, help="Batch size per device.")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning rate.")

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight decay.")

    parser.add_argument("-n", "--max_steps", type=int, default=250, help="Maximum number of steps.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", model_max_length=args.seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_dataset(args.dataset_name, args.dataset_config_name)
    dataset = encode_dataset(dataset, tokenizer)

    config = OPTConfig(
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=args.seq_len,
        num_attention_heads=12,
        vocab_size=50272,
        eos_token_id=1,
        pad_token_id=3,
    )
    model = OPTForCausalLM(config=config)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    training_args = TrainingArguments(
        "hf-opt",
        evaluation_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
    )
    trainer = HfTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
