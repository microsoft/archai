# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.search_spaces.transformer_flex.models.gpt2_flex.configuration_gpt2_flex import (
    GPT2FlexConfig,
)
from archai.nlp.search_spaces.transformer_flex.models.gpt2_flex.modeling_gpt2_flex import (
    GPT2FlexLMHeadModel,
)
from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

if __name__ == "__main__":
    training_args = NvidiaTrainingArguments(
        "gpt2-flex",
        seed=5,
        logging_steps=50,
        eval_steps=1000,
        dataset="wt103",
        vocab="gpt2",
        vocab_size=10000,
        global_batch_size=256,
        seq_len=192,
        strategy="ddp",
        max_steps=1000,
        gradient_accumulation_steps=1,
        optim="adam",
        learning_rate=0.001,
        lr_scheduler_warmup_steps=50,
        lr_scheduler_min_lr=1e-5,
    )

    config = GPT2FlexConfig(
        vocab_size=10000,
        n_positions=192,
        n_embd=768,
        n_layer=5,
        n_head=[4, 4, 8, 8, 8],
        n_inner=[1885, 2005, 2005, 1885, 1885],
        resid_pdrop=0.01,
        embd_pdrop=0.0,
        attn_pdrop=0.01,
        use_cache=False,
        primer_square=True,
    )
    model = GPT2FlexLMHeadModel(config=config)

    trainer = NvidiaTrainer(model=model, args=training_args)
    trainer.train()
