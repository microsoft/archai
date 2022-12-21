# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
)

from archai.nlp.datasets.hf.loaders import encode_dataset, load_dataset
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.quantization import prepare_with_qat
from archai.nlp.trainers.hf.trainer import HfTrainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained("gpt2", model_max_length=192)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataset = encode_dataset(dataset, tokenizer)

    config = GPT2Config(
        vocab_size=50257 + 1,
        n_positions=192,
        n_embd=512,
        n_layer=16,
        n_head=8,
    )
    model = GPT2LMHeadModel(config=config)
    prepare_with_qat(model, onnx_compatible=True)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    training_args = TrainingArguments(
        "hf-qat-gpt2",
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
