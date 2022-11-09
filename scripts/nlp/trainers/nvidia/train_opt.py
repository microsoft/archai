# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM

if __name__ == "__main__":
    training_args = NvidiaTrainingArguments(
        "opt",
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

    config = OPTConfig(
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=16,
        ffn_dim=2048,
        max_position_embeddings=192,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=8,
        use_cache=False
    )
    model = OPTForCausalLM(config=config)

    trainer = NvidiaTrainer(
        model=model,
        args=training_args
    )
    trainer.train()
