# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
)

from archai.common.file_utils import check_available_checkpoint
from archai.datasets.nlp.hf_dataset_provider import HfDiskDatasetProvider

# from archai.datasets.nlp.hf_dataset_provider import HfHubDatasetProvider
# from archai.datasets.nlp.hf_dataset_provider_utils import tokenize_contiguous_dataset
from archai.trainers.nlp.hf_trainer import HfTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a Pareto architecture from Transformer-Flex.")

    parser.add_argument("pareto_config_path", type=str, help="Path to the Pareto architecture configuration file.")

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Defines an output folder for the saved outputs.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=1024)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Users can use the Hugging Face's Hub to download the dataset instead
    # of downloading it manually
    # dataset_provider = HfHubDatasetProvider(dataset="the_pile", subset="plain_text")
    # train_dataset = dataset_provider.get_train_dataset()
    # encoded_train_dataset = train_dataset.map(
    #     tokenize_contiguous_dataset,
    #     batched=True,
    #     fn_kwargs={"tokenizer": tokenizer, "model_max_length": 1024},
    #     remove_columns=train_dataset.column_names,
    # )

    # We pre-encoded the dataset to speed up the training
    dataset_provider = HfDiskDatasetProvider("data/the_pile_gpt2_encoded_1024")
    encoded_train_dataset = dataset_provider.get_train_dataset()
    encoded_val_dataset = dataset_provider.get_val_dataset()
    encoded_test_dataset = dataset_provider.get_test_dataset()

    pareto_config = {}
    with open(args.pareto_config_path, "r") as f:
        pareto_config = json.load(f)

    config = GPT2Config(n_positions=1024, bos_token_id=0, eos_token_id=0, **pareto_config)
    model = GPT2LMHeadModel(config=config)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    training_args = TrainingArguments(
        args.output_dir,
        optim="adamw_torch",
        evaluation_strategy="no",
        logging_steps=10,
        per_device_train_batch_size=64,
        learning_rate=6e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        warmup_steps=150,
        max_steps=30000,
    )
    trainer = HfTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_val_dataset,
    )

    resume_from_checkpoint = check_available_checkpoint(args.output_dir)
    trainer_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_metrics("train", trainer_output.metrics)
    for log_metric in trainer.state.log_history[::-1]:
        if "eval_loss" in log_metric:
            trainer.save_metrics("eval", log_metric)
            break

    test_metric = trainer.evaluate(encoded_test_dataset, metric_key_prefix="test")
    trainer.save_metrics("test", test_metric)
