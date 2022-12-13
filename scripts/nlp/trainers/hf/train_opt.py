# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    OPTConfig,
    OPTForCausalLM,
    TrainingArguments,
)

from archai.nlp.datasets.hf.loaders import encode_dataset, load_dataset
from archai.nlp.trainers.hf.trainer import HfTrainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", model_max_length=192)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataset = encode_dataset(dataset, tokenizer)

    config = OPTConfig(
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
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
        eval_steps=250,
        logging_steps=10,
        per_device_train_batch_size=64,
        learning_rate=0.01,
        weight_decay=0.0,
        max_steps=250,
    )
    trainer = HfTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
