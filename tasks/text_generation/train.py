# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import (
    AutoTokenizer,
    CodeGenConfig,
    CodeGenForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from archai.datasets.nlp.hf_dataset_provider import HfHubDatasetProvider
from archai.datasets.nlp.hf_dataset_provider_utils import tokenize_contiguous_dataset
from archai.trainers.nlp.hf_trainer import HfTrainer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", model_max_length=192)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset_provider = HfHubDatasetProvider(dataset="wikitext", subset="wikitext-103-raw-v1")
    train_dataset = dataset_provider.get_train_dataset()
    eval_dataset = dataset_provider.get_val_dataset()

    encoded_train_dataset = train_dataset.map(
        tokenize_contiguous_dataset,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "model_max_length": 192},
        remove_columns=train_dataset.column_names,
    )
    encoded_eval_dataset = eval_dataset.map(
        tokenize_contiguous_dataset,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "model_max_length": 192},
        remove_columns=eval_dataset.column_names,
    )

    config = CodeGenConfig(
        n_positions=192,
        n_embd=768,
        n_layer=12,
        n_head=12,
        rotary_dim=16,
        bos_token_id=0,
        eos_token_id=0,
        vocab_size=50295,
    )
    model = CodeGenForCausalLM(config=config)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    training_args = TrainingArguments(
        "hf-codegen",
        evaluation_strategy="steps",
        logging_steps=10,
        eval_steps=125,
        per_device_train_batch_size=32,
        learning_rate=0.01,
        weight_decay=0.0,
        max_steps=250,
    )
    trainer = HfTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_eval_dataset,
    )

    trainer.train()
