# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.nlp.datasets.hf.loaders import load_dataset, encode_dataset
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import ArchaiPreTrainedTokenizerFast
from archai.nlp.trainers.hf.trainer import HfTrainer

from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.training_args import TrainingArguments
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained("gpt2-large")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataset = encode_dataset(tokenizer, dataset)

    config = GPT2Config(
        vocab_size=50257+1,
        n_positions=1024,
        n_embd=512,
        n_layer=4,
        n_head=8,
    )
    model = GPT2LMHeadModel(config=config)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    training_args = TrainingArguments(
        "gpt2",
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        max_steps=10,
    )
    trainer = HfTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
