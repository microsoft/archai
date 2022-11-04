# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.trainers.nvidia.trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia.training_args import NvidiaTrainingArguments

from archai.nlp.search_spaces.transformer_flex.models.gpt2_flex.config_gpt2_flex import GPT2FlexConfig
from archai.nlp.search_spaces.transformer_flex.models.gpt2_flex.model_gpt2_flex import GPT2FlexLMHeadModel

if __name__ == "__main__":
    training_args = NvidiaTrainingArguments(
        "gpt2-flex",
        seed=5,
        use_cuda=True,
        log_interval=50,
        eval_interval=1000,
        dataset="wt103",
        vocab="gpt2",
        vocab_size=10000,
        batch_size=256,
        seq_len=192,
        strategy="ddp",
        max_steps=1000,
        gradient_accumulation_steps=1,
        optimizer="adam",
        optimizer_lr=0.001,
        scheduler_warmup_steps=50,
        scheduler_lr_min=1e-5,
    )

    config = GPT2FlexConfig(
        vocab_size=10000,
        n_positions=192,
        n_embd=256,
        n_layer=5,
        n_head=8,
        n_inner=1885,
        resid_pdrop=0.01,
        embd_pdrop=0.0,
        attn_pdrop=0.01,
        use_cache=False,
        primer_square=True,
    )
    model = GPT2FlexLMHeadModel(config=config)

    trainer = NvidiaTrainer(
        model=model,
        args=training_args
    )
    trainer.train()
