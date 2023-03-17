# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from transformers import (
    AutoTokenizer,
    CodeGenConfig,
    CodeGenForCausalLM,
    TrainingArguments,
)

from archai.datasets.nlp.fast_hf_dataset_provider import (
    FastDataCollatorForLanguageModeling,
    FastHfDatasetProvider,
)
from archai.trainers.nlp.hf_trainer import HfTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a CodeGen model using the Hugging Face trainer.")

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

    parser.add_argument("-ls", "--logging_steps", type=int, default=10, help="Number of steps between logs.")

    parser.add_argument("-es", "--eval_steps", type=int, default=100, help="Number of steps between evaluations.")

    parser.add_argument("-ss", "--save_steps", type=int, default=100, help="Number of steps between checkpoints.")

    parser.add_argument("-bsz", "--per_device_train_batch_size", type=int, default=64, help="Batch size per device.")

    parser.add_argument("-n", "--max_steps", type=int, default=1, help="Maximum number of steps.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    collator = FastDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset_provider = FastHfDatasetProvider.from_hub(
        args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        tokenizer=tokenizer,
    )

    train_dataset = dataset_provider.get_train_dataset(seq_len=2048)
    eval_dataset = dataset_provider.get_val_dataset(seq_len=2048)

    config = CodeGenConfig(
        vocab_size=50295,
        n_positions=2048,
        n_embd=1024,
        n_layer=20,
        n_head=16,
        rotary_dim=32,
    )
    model = CodeGenForCausalLM(config=config)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    training_args = TrainingArguments(
        "hf-codegen",
        evaluation_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=1.8e-3,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
    )
    trainer = HfTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
