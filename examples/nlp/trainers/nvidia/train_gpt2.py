# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import GPT2Config, GPT2LMHeadModel

from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

if __name__ == "__main__":
    training_args = NvidiaTrainingArguments(
        "gpt2",
        seed=42,
        logging_steps=10,
        eval_steps=100,
        dataset="wt103",
        vocab="gpt2",
        vocab_size=10000,
        global_batch_size=256,
        seq_len=192,
        strategy="ddp",
        max_steps=250,
        gradient_accumulation_steps=1,
        optim="jitlamb",
    )

    config = GPT2Config(
        vocab_size=10000,
        n_positions=192,
        n_embd=512,
        n_layer=16,
        n_head=8,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config=config)

    trainer = NvidiaTrainer(model=model, args=training_args)
    trainer.train()
