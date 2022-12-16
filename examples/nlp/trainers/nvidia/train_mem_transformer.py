# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.configuration_mem_transformer import (
    MemTransformerConfig,
)
from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.modeling_mem_transformer import (
    MemTransformerLMHeadModel,
)
from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

if __name__ == "__main__":
    training_args = NvidiaTrainingArguments(
        "mem-transformer",
        seed=37,
        logging_steps=50,
        eval_steps=1000,
        dataset="wt103",
        vocab="word",
        vocab_size=267735,
        global_batch_size=256,
        seq_len=192,
        strategy="ddp",
        max_steps=1000,
        gradient_accumulation_steps=4,
        optim="adam",
        learning_rate=0.001,
        lr_scheduler_warmup_steps=50,
        lr_scheduler_min_lr=1e-5,
        find_unused_parameters=True,
    )

    config = MemTransformerConfig(
        vocab_size=267735,
        n_positions=192,
        d_model=256,
        d_embd=256,
        n_head=8,
        d_head=32,
        d_inner=1024,
        div_val=4,
        n_layer=6,
        mem_len=0,
        dropout=0.01,
        dropatt=0.01,
        use_cache=False,
        primer_square=True,
    )
    model = MemTransformerLMHeadModel(config=config)

    trainer = NvidiaTrainer(model=model, args=training_args)
    trainer.train()
