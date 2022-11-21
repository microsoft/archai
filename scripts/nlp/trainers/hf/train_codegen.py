# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import (
    CodeGenConfig,
    CodeGenForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from archai.nlp.datasets.hf.loaders import encode_dataset, load_dataset
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.trainers.hf.trainer import HfTrainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono", model_max_length=192)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataset = encode_dataset(dataset, tokenizer)

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
