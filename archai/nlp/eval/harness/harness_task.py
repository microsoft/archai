# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-based task.
"""

from __future__ import annotations

import importlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
from datasets.arrow_dataset import Dataset
from evaluate import load as hf_load_metric

from archai.common.utils import cached_property
from archai.nlp.datasets.hf.loaders import load_dataset
from archai.nlp.eval.harness.harness_utils import HarnessCall

datasets.disable_progress_bar()

AVAILABLE_HARNESS_TASKS = {
    "arc_challenge": "ARCChallengeHarnessTask",
    "arc_easy": "ARCEasyHarnessTask",
    "axb": "AXbHarnessTask",
    "axg": "AXgHarnessTask",
    "boolq": "BoolQHarnessTask",
    "cb": "CBHarnessTask",
    "cola": "CoLAHarnessTask",
    "copa": "COPAHarnessTask",
    "hella_swag": "HellaSwagHarnessTask",
    "human_eval": "HumanEvalHarnessTask",
    "mnli_matched": "MNLIMatchedHarnessTask",
    "mnli_mismatched": "MNLIMismatchedHarnessTask",
    "mrpc": "MRPCHarnessTask",
    "multirc": "MultiRCHarnessTask",
    "open_book_qa": "OpenBookQAHarnessTask",
    "piqa": "PIQAHarnessTask",
    "qnli": "QNLIHarnessTask",
    "qqp": "QQPHarnessTask",
    "record": "ReCoRDHarnessTask",
    "rte": "RTEHarnessTask",
    "sst2": "SST2HarnessTask",
    "stsb": "STSBHarnessTask",
    "story_cloze_2016": "StoryCloze2016HarnessTask",
    "story_cloze_2018": "StoryCloze2018HarnessTask",
    "wic": "WiCHarnessTask",
    "wino_grande": "WinoGrandeHarnessTask",
    "wnli": "WNLIHarnessTask",
    "wsc": "WSCHarnessTask",
}


def load_harness_task(task_name: str, **kwargs) -> HarnessTask:
    """Instantiates a new harness task.

    Args:
        task_name: Name of harness task to be instantiated.

    Returns:
        (HarnessTask): A harness task wrapped into corresponding class.

    """

    available_tasks = list(AVAILABLE_HARNESS_TASKS.keys())
    assert task_name in available_tasks, f"`task_name` should be in {available_tasks}."
    task_cls_name = AVAILABLE_HARNESS_TASKS[task_name]

    task_module = importlib.import_module("archai.nlp.eval_utils.harness.tasks")
    task_cls = getattr(task_module, task_cls_name)

    return task_cls(**kwargs)


class HarnessTask:
    """Implements a harness-based task."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
        metric_name: Optional[str] = None,
        metric_config_name: Optional[str] = None,
    ) -> None:
        """Initializes with custom arguments and keyword arguments.

        Args:
            dataset_name: Name of dataset to be downloaded.
            dataset_config_name: Name of configuration of dataset to be downloaded.
            dataset_dir: Path to manually downloaded files.
            dataset_split: Split to be retrieved. `None` defaults to all splits.
            dataset_cache: Folder where cache should be stored/loaded.
            dataset_samples: Subsamples into a fixed amount of samples.
            random_seed: Fixes the order of samples.
            num_proc: Number of processes for multiprocessing.
            metric_name: Name of metric to be instantiated.
            metric_config_name: Configuration name of metric to be instantiated.

        """

        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.dataset_dir = dataset_dir
        self.dataset_split = dataset_split
        self.dataset_samples = dataset_samples
        self.random_seed = random_seed
        self.num_proc = num_proc
        self.metric_name = metric_name
        self.metric_config_name = metric_config_name

        self.dataset = load_dataset(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            dataset_dir=dataset_dir,
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            n_samples=dataset_samples,
            random_seed=random_seed,
        )

        if not metric_name:
            metric_name = "accuracy"
        self.metric = hf_load_metric(metric_name, metric_config_name)

    @property
    def has_train_set(self) -> bool:
        """Whether task has an available training set.

        Returns:
            (bool): Availability of training set.

        """

        return bool("train" in self.dataset)

    @cached_property
    def train_set(self) -> Dataset:
        """Training set.

        Returns:
            (Dataset): Training set with `_pre_process_sample` function applied.

        """

        if not self.has_train_set:
            return []

        train_set = self.dataset["train"]
        train_set = train_set.map(self._pre_process_sample, num_proc=self.num_proc)

        return train_set

    @property
    def has_validation_set(self) -> bool:
        """Whether task has an available validation set.

        Returns:
            (bool): Availability of validation set.

        """

        return bool("validation" in self.dataset)

    @cached_property
    def validation_set(self) -> Dataset:
        """Validation set.

        Returns:
            (Dataset): Validation set with `_pre_process_sample` function applied.

        """

        if not self.has_validation_set:
            return []

        validation_set = self.dataset["validation"]
        validation_set = validation_set.map(self._pre_process_sample, num_proc=self.num_proc)

        return validation_set

    @property
    def has_test_set(self) -> bool:
        """Whether task has an available testing set.

        Returns:
            (bool): Availability of testing set.

        """

        return bool("test" in self.dataset)

    @cached_property
    def test_set(self) -> Dataset:
        """Testing set.

        Returns:
            (Dataset): Testing set with `_pre_process_sample` function applied.

        """

        if not self.has_test_set:
            return []

        test_set = self.dataset["test"]
        test_set = test_set.map(self._pre_process_sample, num_proc=self.num_proc)

        return test_set

    @property
    def config(self) -> Dict[str, Any]:
        """Task's configuration.

        Returns:
            (Dict[str, Any]): Information about the task.

        """

        return {
            "dataset": {
                "name": self.dataset_name,
                "config_name": self.dataset_config_name,
                "dataset_samples": self.dataset_samples,
                "has_train_set": self.has_train_set,
                "has_validation_set": self.has_validation_set,
                "has_test_set": self.has_test_set,
                "random_seed": self.random_seed,
            },
            "metric": {"name": self.metric_name, "config_name": self.metric_config_name},
        }

    def _pre_process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-processes a dataset by using a map function.

        Args:
            samples: Incoming samples.

        Returns:
            (List[Dict[str, Any]]): Pre-processed samples.

        """

        return sample

    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        """Creates the string-based content of sample.

        This function needs to be overidden by childs due to different
        data format of tasks.

        Args:
            sample: Sample.

        Returns:
            (str): String-based content of sample.

        """

        raise NotImplementedError

    def _create_label(self, sample: Dict[str, Any]) -> str:
        """Creates the string-based label of sample.

        This function needs to be overidden by childs due to different
        data format of tasks.

        Args:
            sample: Sample.

        Returns:
            (str): String-based label of sample.

        """

        raise NotImplementedError

    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        """Creates a tuple of HarnessCall that runs the sampling procedure.

        This function needs to be overidden by childs due to different
        sampling of tasks.

        Args:
            sample: Sample.
            context: Context.

        Returns:
            (Tuple[HarnessCall, ...]): Sampling procedures.

        """

        raise NotImplementedError

    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        """Computes a tuple of results from the sampling procedure and
            adds current prediction/reference to metric.

        This function needs to be overidden by childs due to different
        post-processing of tasks.

        Args:
            sample: Sample.
            results: Results.

        """

        raise NotImplementedError

    def compute_metrics(self) -> Dict[str, Any]:
        """Computes metrics from stored predictions and references.

        Returns:
            (Dict[str, Any]): Dictionary of metrics.

        """

        return self.metric.compute()

    def _get_few_shot_samples(self, sample_set: Dataset, k: int) -> List[Dict[str, Any]]:
        """Gets `k` few-shot samples.

        Args:
            sample_set: Set to be sampled from.
            k: Number of few-shot samples.

        Returns:
            (List[Dict[str, Any]]): Few-shot samples.

        """

        # Ensures that `k` is a valid number
        if k > sample_set.num_rows or k < 0:
            return []

        sample_idx = random.sample(range(sample_set.num_rows), k)

        return sample_set.select(sample_idx)

    def create_context(
        self,
        sample: Dict[str, Any],
        n_few_shot: Optional[int] = 0,
        description: Optional[str] = None,
    ) -> str:
        """Creates a context based on current evaluation sample and few-shot samples.

        Args:
            sample: Current evaluation sample.
            n_few_shot: Number of few-shot samples.
            description: Additional description to be added to the context.

        Returns:
            (str): String-based context.

        """

        description = description + "\n\n" if description else ""
        inputs = self._create_inputs(sample)

        # For zero-shot, no context is available
        if n_few_shot == 0:
            return description + inputs

        # For few-shot, `n` samples are retrieved from the training set
        context_samples = []
        if self.has_train_set:
            context_samples = self._get_few_shot_samples(self.train_set, n_few_shot)

        # If the training set is not available, the samples should
        # be retrieved from validation or test set, with the caveat
        # that they need to be different from current sample
        if self.has_validation_set or self.has_test_set:
            context_samples = self._get_few_shot_samples(self.validation_set or self.test_set, n_few_shot + 1)
            context_samples = [c_sample for c_sample in context_samples if c_sample != sample][:n_few_shot]

        # Creates the context by joining inputs and labels from the few-shot samples
        context = (
            "\n\n".join([self._create_inputs(c_sample) + self._create_label(c_sample) for c_sample in context_samples])
            + "\n\n"
        )

        return description + context + inputs
