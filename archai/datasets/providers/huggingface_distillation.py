# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional

from overrides import overrides, EnforceOverrides
import torch
from torch.utils.data.dataset import Dataset

from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    set_seed,
    GPT2TokenizerFast, GPT2Config,
    PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizerBase
)

import torch
from torch.utils.data import TensorDataset, DataLoader

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils
from archai.nlp.data_training_arguments import DataTrainingArguments



class HuggingFaceDistillationProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config, teacher_model:PreTrainedModel):
        super().__init__(conf_dataset)
        self.conf_dataset = conf_dataset
        self.teacher_model = teacher_model
        self._dataroot = utils.full_path(conf_dataset['dataroot'])


    def get_hf_datasets(self, data_args:DataTrainingArguments)->DatasetDict:
        if data_args.dataset_name is not None:
            datasets = HuggingFaceDistillationProvider.dataset_from_name(data_args.dataset_name,
                                        data_args.dataset_config_name, data_args.data_dir,
                                        data_args.validation_split_percentage)
        elif data_args.train_file is not None:
            datasets = HuggingFaceDistillationProvider.dataset_from_files(data_args.train_file, data_args.validation_file)
        else:
            raise ValueError('Either dataset_name or train_file must be provided')

        assert isinstance(datasets, DatasetDict)
        return datasets

    @staticmethod
    def dataset_from_files(train_file:Optional[str], validation_file:Optional[str])->DatasetDict:
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
        if validation_file is not None:
            data_files["validation"] = validation_file
        extension = (
            train_file.split(".")[-1]
            if train_file is not None
            else validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)

        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        assert isinstance(datasets, DatasetDict)
        return datasets

    @staticmethod
    def dataset_from_name(dataset_name:str, dataset_config_name:Optional[str],
                        data_dir:Optional[str], validation_split_percentage:Optional[int])->DatasetDict:

        datasets = load_dataset(dataset_name, dataset_config_name,
                                cache_dir=data_dir)
        assert isinstance(datasets, DatasetDict)

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{validation_split_percentage}%]",
                cache_dir=data_dir
            )
            datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{validation_split_percentage}%:]",
                cache_dir=data_dir
            )

        return datasets

    def soft_labels_ds(self, dataset:Dataset, teacher_model:PreTrainedModel)->Dataset:
        inputs, soft_labels = [], []
        for input_ids, labels in dataset:
            preds = teacher_model(input_ids)
            next_token_logits = preds[0][:, -1, :]
            inputs.append(input_ids)
            soft_labels.append(next_token_logits)

        return TensorDataset(torch.Tensor(inputs), torch.Tensor(soft_labels))

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        data_args = DataTrainingArguments(self.conf_dataset)

        hf_datasets = self.get_hf_datasets(data_args)

        train_soft_ds = self.soft_labels_ds(hf_datasets['train'], self.teacher_model)
        test_soft_ds = self.soft_labels_ds(hf_datasets['test'], self.teacher_model)

        return train_soft_ds, test_soft_ds


register_dataset_provider('huggingface_distillation', HuggingFaceDistillationProvider)