# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import OPTConfig, OPTForCausalLM

from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

if __name__ == "__main__":
    training_args = NvidiaTrainingArguments(
        "opt",
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

    config = OPTConfig(
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=16,
        ffn_dim=2048,
        max_position_embeddings=192,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=8,
        use_cache=False,
    )
    model = OPTForCausalLM(config=config)

    trainer = NvidiaTrainer(model=model, args=training_args)
    trainer.train()
