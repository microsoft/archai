import os
from typing import Optional, Union

from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.training_args import TrainingArguments

from archai.discrete_search.search_spaces.config import ArchConfig
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import ArchaiPreTrainedTokenizerFast
from archai.nlp.trainers.hf.callbacks import PerplexityTrainerCallback
from archai.nlp.trainers.hf.trainer import HfTrainer
from archai.nlp.datasets.hf.loaders import load_dataset, encode_dataset

from transformer_plus_plus.search_space.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from training.utils import from_yaml_file, from_json_file, group_texts


class Experiment:
    def __init__(
        self,
        arch_config: Union[str, os.PathLike],
        experiment_config: Union[str, os.PathLike],
        output_dir: Optional[str] = "",
    ) -> None:
        config = from_yaml_file(experiment_config)

        data_config = config.get("data", {}) or {}
        tokenizer_config = config.get("tokenizer", {}) or {}

        trainer_config = config.get("trainer", {}) or {}
        if trainer_config.get("fp16", None) is None:
            trainer_config["fp16"] = False

        model_config = config.get("model", {}) or {}
        model_config["fp16"] = trainer_config["fp16"]

        self.arch_config = arch_config
        self.config = config
        self.output_dir = os.path.abspath(output_dir)

        self.dataset_config = data_config.pop("dataset", {}) or {}
        self.grouped = data_config.pop('grouped', False)
        self.data_config = data_config

        self.tokenizer_path = tokenizer_config.pop("tokenizer_path", "gpt2")
        self.tokenizer_config = tokenizer_config

        self.random_seed = trainer_config.get("seed", 0)
        self.trainer_config = trainer_config

        self.model_name = model_config.get("model_name", "")
        self.model_config = model_config

    @property
    def results_dir(self) -> str:
        return os.path.join(self.output_dir, self.model_name)

    def run(self, resume_from_checkpoint: Optional[Union[str, bool]] = None) -> None:
        training_args = TrainingArguments(
            self.results_dir,
            **self.trainer_config,
        )

        tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained(
            self.tokenizer_path,
            **self.tokenizer_config
        )
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        with training_args.main_process_first(desc="loading and preparing dataset"):
            dataset = load_dataset(**self.dataset_config, random_seed=self.random_seed)
            
            if not self.grouped:
                dataset = encode_dataset(dataset, tokenizer, **self.data_config)
            else:
                def tokenize_function(examples):
                    return tokenizer(examples['text'])

                dataset = dataset.map(
                    tokenize_function, batched=True, num_proc=4,
                    remove_columns=['text'], desc='Tokenizing...'
                )

                dataset = dataset.map(
                    group_texts, batched=True, fn_kwargs={'tokenizer': tokenizer},
                    num_proc=4, desc='Grouping...'
                )


        hf_config = GPT2Config(**self.model_config)
        arch_config = ArchConfig.from_json(str(self.arch_config))
        model = GPT2LMHeadModel(arch_config, hf_config)

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
