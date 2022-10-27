# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from json import load
from operator import mod
from archai.nlp.search_spaces.transformer_flex.models.model_loader import load_model_from_config
from archai.nlp.trainers.nvidia_trainer import NvidiaTrainer
from archai.nlp.trainers.nvidia_training_args import NvidiaTrainingArguments

if __name__ == "__main__":
    model = load_model_from_config("hf_gpt2", {})
    print(model)

    training_args = NvidiaTrainingArguments("gpt2", use_cuda=False)
    trainer = NvidiaTrainer(
        model=model,
        args=training_args
    )
