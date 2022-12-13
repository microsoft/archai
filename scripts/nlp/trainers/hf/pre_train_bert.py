# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import (
    BertConfig,
    BertForPreTraining,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from archai.nlp.datasets.hf.loaders import encode_dataset, load_dataset
from archai.nlp.datasets.hf.processors import tokenize_nsp_dataset
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.trainers.hf.trainer import HfTrainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=192)
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataset = encode_dataset(dataset, tokenizer, mapping_fn=tokenize_nsp_dataset)

    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        pad_token_id=3,
        vocab_size=30522,
    )
    model = BertForPreTraining(config=config)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    training_args = TrainingArguments(
        "hf-bert",
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
