# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-based task."""

from __future__ import annotations

import importlib
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
from datasets.arrow_dataset import Dataset
from evaluate import load as hf_load_metric
from overrides.enforce import EnforceOverrides

from archai.nlp.datasets.hf.loaders import load_dataset
from archai.nlp.eval.eval_utils import cached_property
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
    "lambada": "LambadaHarnessTask",
    "lambada_openai": "LambadaOpenAIHarnessTask",
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
    """Instantiate a new harness task of the specified type.

    This function loads a harness task class based on the provided `task_name`
    and creates an instance of the class with the provided keyword arguments.

    Args:
        task_name: The name of the harness task to instantiate.

    Returns:
        An instance of the specified harness task class.

    Raises:
        AssertionError: If the provided `task_name` is not a valid key
            in the `AVAILABLE_HARNESS_TASKS` dictionary.

    """

    available_tasks = list(AVAILABLE_HARNESS_TASKS.keys())
    assert task_name in available_tasks, f"`task_name` should be in {available_tasks}."
    task_cls_name = AVAILABLE_HARNESS_TASKS[task_name]

    task_module = importlib.import_module("archai.nlp.eval.harness.tasks")
    task_cls = getattr(task_module, task_cls_name)

    return task_cls(**kwargs)


class HarnessTask(EnforceOverrides):
    """Abstract base class for harness-based tasks.

    A harness-based task is a task that involves evaluating a model's performance
    on a particular task using a dataset, a metric, and possibly some pre-processing steps.

    HarnessTask subclasses must override the `_create_inputs`, `_create_label`,
    `create_sampling_calls` and `compute_results` method.

    """

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
        """Initialize a new harness task.

        Args:
            dataset_name: The name of the dataset to load.
            dataset_config_name: The name of the configuration for the dataset to load.
            dataset_dir: The directory where the dataset files can be found.
            dataset_split: The splits of the dataset to load.
            dataset_samples: The number of samples to subsample from the dataset.
            random_seed: The seed used to shuffle the samples in the dataset.
            num_proc: The number of processes to use for multiprocessing
                when preparing the data.
            metric_name: The name of the metric to instantiate.
            metric_config_name: The name of the configuration for the metric to instantiate.

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
            Availability of training set.

        """

        return bool("train" in self.dataset)

    @cached_property
    def train_set(self) -> Dataset:
        """Training set.

        Returns:
            The training set, with the `_pre_process_sample` function applied.

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
            Availability of validation set.

        """

        return bool("validation" in self.dataset)

    @cached_property
    def validation_set(self) -> Dataset:
        """Validation set.

        Returns:
            The validation set, with the `_pre_process_sample` function applied.

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
            Availability of testing set.

        """

        return bool("test" in self.dataset)

    @cached_property
    def test_set(self) -> Dataset:
        """Testing set.

        Returns:
            The testing set, with the `_pre_process_sample` function applied.

        """

        if not self.has_test_set:
            return []

        test_set = self.dataset["test"]
        test_set = test_set.map(self._pre_process_sample, num_proc=self.num_proc)

        return test_set

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration of the task.

        Returns:
            A dictionary containing the configuration of the task, including information about
                the dataset and metric used.

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
        """Pre-process a sample.

        Args:
            sample: The input sample.

        Returns:
            The pre-processed sample.

        """

        return sample

    @abstractmethod
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        """Create the string-based content of the sample.

        This function needs to be overidden by children due to different
        data format of tasks.

        Args:
            sample: The input sample.

        Returns:
            String-based content of sample.

        """

    @abstractmethod
    def _create_label(self, sample: Dict[str, Any]) -> str:
        """Create the string-based label of the sample.

        This function needs to be overidden by children due to different
        data format of tasks.

        Args:
            sample: The input sample.

        Returns:
            String-based label of sample.

        """

    @abstractmethod
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        """Create a tuple of `HarnessCall` that runs the sampling procedure.

        This function needs to be overidden by children due to different
        sampling of tasks.

        Args:
            sample: The input sample.
            context: The context of the sample.

        Returns:
            Tuple of `HarnessCall` that runs the sampling procedure.

        """

    @abstractmethod
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        """Computes a tuple of results from the sampling procedure and
        add current prediction/reference to metric.

        This function needs to be overidden by children due to different
        post-processing of tasks.

        Args:
            sample: The input sample.
            results: A tuple of results from the sampling procedure.

        """

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute metrics from stored predictions and references.

        Returns:
            Dictionary of metrics.

        """

        return self.metric.compute()

    def _get_few_shot_samples(self, sample_set: Dataset, k: int) -> List[Dict[str, Any]]:
        """Get `k` few-shot samples.

        Args:
            sample_set: Dataset of few-shot samples.
            k: Number of few-shot samples.

        Returns:
            List of few-shot samples.

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
        """Create a context based on current evaluation sample and few-shot samples.

        Args:
            sample: The input sample.
            n_few_shot: Number of few-shot samples.
            description: An additional description to be added to the context.

        Returns:
            String-based context of sample.

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
