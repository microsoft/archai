# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

if __name__ == "__main__":
    training_args = NvidiaTrainingArguments(
        "gpt2",
        seed=42,
        use_cuda=True,
        log_interval=10,
        eval_interval=100,
        dataset="wt103",
        vocab="gpt2",
        vocab_size=10000,
        batch_size=256,
        seq_len=192,
        strategy="ddp",
        max_steps=250,
        gradient_accumulation_steps=1,
        optimizer="jitlamb",
    )

    config = GPT2Config(
        vocab_size=10000,
        n_positions=192,
        n_embd=512,
        n_layer=16,
        n_head=8,
    )
    model = GPT2LMHeadModel(config=config)
    
    trainer = NvidiaTrainer(
        model=model,
        args=training_args
    )
    trainer.train()
