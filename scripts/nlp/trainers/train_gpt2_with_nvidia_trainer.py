# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# from archai.nlp.search_spaces.transformer_flex.models.model_loader import load_model_from_config
from archai.nlp.trainers.nvidia_trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia_training_args import NvidiaTrainingArguments

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

if __name__ == "__main__":
    # model = load_model_from_config("hf_gpt2", {})
    # print(model)

    config = GPT2Config(
        vocab_size=10000,
        n_positions=192,
        n_embd=512,
        n_layer=4,
        n_head=8,
    )
    model = GPT2LMHeadModel(config=config)

    training_args = NvidiaTrainingArguments("gpt2", use_cuda=False)
    trainer = NvidiaTrainer(
        model=model,
        args=training_args
    )

    trainer.train()
