# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Optional, Union

from src.modeling_codegen_conv_att import (
    CodeGenConvAttConfig,
    CodeGenConvAttForCausalLM,
)
from src.modeling_codegen_hard_coded import (
    CodeGenHardCodedConfig,
    CodeGenHardCodedForCausalLM,
)
from src.modeling_codegen_sgconv import CodeGenSGConvConfig, CodeGenSGConvForCausalLM
from src.utils import from_yaml_file, load_collator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from archai.nlp import logging_utils
from archai.nlp.datasets.hf.loaders import encode_dataset, load_dataset
from archai.nlp.datasets.hf.processors import merge_datasets
from archai.nlp.trainers.hf.callbacks import PerplexityTrainerCallback
from archai.nlp.trainers.hf.trainer import HfTrainer

logger = logging_utils.get_logger(__name__)

# Register internal models to be compatible with auto classes
AutoConfig.register("codegen_conv_att", CodeGenConvAttConfig)
AutoConfig.register("codegen_hard_coded", CodeGenHardCodedConfig)
AutoConfig.register("codegen_sgconv", CodeGenSGConvConfig)

AutoModelForCausalLM.register(CodeGenConvAttConfig, CodeGenConvAttForCausalLM)
AutoModelForCausalLM.register(CodeGenHardCodedConfig, CodeGenHardCodedForCausalLM)
AutoModelForCausalLM.register(CodeGenSGConvConfig, CodeGenSGConvForCausalLM)


class Experiment:
    def __init__(
        self,
        experiment_config: Union[str, os.PathLike],
        output_dir: Optional[str] = "",
    ) -> None:
        logger.info(f"Creating experiment: {experiment_config}")

        config = from_yaml_file(experiment_config)

        collator_config = config.get("collator", {}) or {}
        data_config = config.get("data", {}) or {}
        tokenizer_config = config.get("tokenizer", {}) or {}

        trainer_config = config.get("trainer", {}) or {}
        if trainer_config.get("fp16", None) is None:
            trainer_config["fp16"] = False

        model_config = config.get("model", {}) or {}
        model_config["fp16"] = trainer_config["fp16"]

        self.config = config
        self.output_dir = os.path.abspath(output_dir)

        self.collator_name = collator_config.pop("collator_name", "language-modelling")
        self.collator_config = collator_config
        self.dataset_config = data_config.pop("dataset", []) or []
        self.data_config = data_config
        self.tokenizer_config = tokenizer_config

        self.random_seed = trainer_config.get("seed", 0)
        self.optimizer = trainer_config.pop("optimizer", "adamw")
        self.trainer_config = trainer_config

        self.model_type = model_config.pop("model_type", "codegen")
        self.model_config = model_config

        logger.info("Experiment created.")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.output_dir, self.model_type)

    def run(self, resume_from_checkpoint: Optional[Union[str, bool]] = None) -> None:
        logger.info("Running experiment ...")

        training_args = TrainingArguments(
            self.results_dir,
            **self.trainer_config,
        )

        tokenizer_file = self.tokenizer_config.pop("tokenizer_file", "")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_file, **self.tokenizer_config)

        collator = load_collator(self.collator_name, tokenizer=tokenizer, **self.collator_config)

        with training_args.main_process_first(desc="loading and preparing dataset"):
            datasets = [
                load_dataset(
                    **dataset_config,
                    random_seed=self.random_seed,
                )
                for dataset_config in self.dataset_config
            ]
            dataset = merge_datasets(datasets)
            dataset = encode_dataset(dataset, tokenizer, **self.data_config)

        config = AutoConfig.for_model(self.model_type, **self.model_config)
        model = AutoModelForCausalLM.from_config(config=config)

        trainer = HfTrainer(
            model=model,
            args=training_args,
            callbacks=[PerplexityTrainerCallback],
            data_collator=collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else None,
        )

        trainer_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        trainer.save_metrics("train", trainer_output.metrics)
        if "validation" in dataset:
            for log_metric in trainer.state.log_history[::-1]:
                if "eval_loss" in log_metric:
                    trainer.save_metrics("eval", log_metric)
                    break
        if "test" in dataset:
            test_metric = trainer.evaluate(dataset["test"], metric_key_prefix="test")
            trainer.save_metrics("test", test_metric)
