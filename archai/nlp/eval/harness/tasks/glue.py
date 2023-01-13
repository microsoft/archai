# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
https://openreview.net/pdf?id=rJ4km2R5t7
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets.arrow_dataset import Dataset
from overrides import overrides

from archai.nlp.eval.eval_utils import cached_property
from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import (
    HarnessCall,
    call_factory,
    clean_sample_text,
)


class CoLAHarnessTask(HarnessTask):
    """CoLA harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="cola",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="cola",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['sentence']}\nQuestion: Does this sentence make sense?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "no", 1: "yes"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_no = call_factory.log_likelihood(context, " no")
        ll_yes = call_factory.log_likelihood(context, " yes")

        return ll_no, ll_yes

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_no, ll_yes = results

        prediction = int(ll_yes > ll_no)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class MNLIMatchedHarnessTask(HarnessTask):
    """MNLI (matched) harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="mnli",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="mnli",
        )

    @property
    def has_validation_set(self) -> bool:
        return bool("validation_matched" in self.dataset)

    @cached_property
    def validation_set(self) -> Dataset:
        if not self.has_validation_set:
            return []

        validation_set = self.dataset["validation_matched"]
        validation_set = validation_set.map(self._pre_process_sample, num_proc=self.num_proc)

        return validation_set

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        premise = sample["premise"]
        hypothesis = sample["hypothesis"].strip() + ("" if sample["hypothesis"].strip().endswith(".") else ".")

        return f"{premise}\nQuestion: {hypothesis} True, False or Neither?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "True", 1: "Neither", 2: "False"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_true = call_factory.log_likelihood(context, " True")
        ll_neither = call_factory.log_likelihood(context, " Neither")
        ll_false = call_factory.log_likelihood(context, " False")

        return ll_true, ll_neither, ll_false

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = np.argmax(results)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class MNLIMismatchedHarnessTask(MNLIMatchedHarnessTask):
    """MNLI (mismatched) harness task."""

    @property
    def has_validation_set(self) -> bool:
        return bool("validation_mismatched" in self.dataset)

    @cached_property
    def validation_set(self) -> Dataset:
        if not self.has_validation_set:
            return []

        validation_set = self.dataset["validation_mismatched"]
        validation_set = validation_set.map(self._pre_process_sample, num_proc=self.num_proc)

        return validation_set


class MRPCHarnessTask(HarnessTask):
    """MRPC harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="mrpc",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="mrpc",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        sentence1 = clean_sample_text(sample["sentence1"])
        sentence2 = clean_sample_text(sample["sentence2"])

        return f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nQuestion: Do both sentences mean the same thing?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "no", 1: "yes"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_no = call_factory.log_likelihood(context, " no")
        ll_yes = call_factory.log_likelihood(context, " yes")

        return ll_no, ll_yes

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_no, ll_yes = results

        prediction = int(ll_yes > ll_no)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class QNLIHarnessTask(HarnessTask):
    """QNLI harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="qnli",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="qnli",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['question']}\n{sample['sentence']}\nQuestion: Does this response answer the question?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "yes", 1: "no"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_yes = call_factory.log_likelihood(context, " yes")
        ll_no = call_factory.log_likelihood(context, " no")

        return ll_yes, ll_no

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_yes, ll_no = results

        prediction = int(ll_no > ll_yes)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class QQPHarnessTask(HarnessTask):
    """QQP harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="qqp",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="qqp",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"Question 1: {sample['question1']}\nQuestion 2: {sample['question2']}\nQuestion: Do both questions ask the same thing?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "no", 1: "yes"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_no = call_factory.log_likelihood(context, " no")
        ll_yes = call_factory.log_likelihood(context, " yes")

        return ll_no, ll_yes

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_no, ll_yes = results

        prediction = int(ll_yes > ll_no)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class RTEHarnessTask(HarnessTask):
    """RTE harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="rte",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="rte",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['sentence1']}\nQuestion: {sample['sentence2']} True or False?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "True", 1: "False"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_true = call_factory.log_likelihood(context, " True")
        ll_false = call_factory.log_likelihood(context, " False")

        return ll_true, ll_false

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_true, ll_false = results

        prediction = int(ll_false > ll_true)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class SST2HarnessTask(HarnessTask):
    """SST-2 harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="sst2",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="sst2",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{clean_sample_text(sample['sentence'])}\nQuestion: Is this sentence positive or negative?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "negative", 1: "positive"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_negative = call_factory.log_likelihood(context, " negative")
        ll_positive = call_factory.log_likelihood(context, " positive")

        return ll_negative, ll_positive

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_negative, ll_positive = results

        prediction = int(ll_positive > ll_negative)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class STSBHarnessTask(HarnessTask):
    """STS-B harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="stsb",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="stsb",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return ""

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        return ""

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        return call_factory.cosine_similarity(sample["sentence1"], sample["sentence2"])

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = results[0]
        reference = sample["label"] / 5

        self.metric.add(predictions=prediction, reference=reference)


class WNLIHarnessTask(HarnessTask):
    """WNLI harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "glue",
            dataset_config_name="wnli",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="glue",
            metric_config_name="wnli",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['sentence1']}\nQuestion: {sample['sentence2']} True or False?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "False", 1: "True"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_false = call_factory.log_likelihood(context, " False")
        ll_true = call_factory.log_likelihood(context, " True")

        return ll_false, ll_true

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_false, ll_true = results

        prediction = int(ll_true > ll_false)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)
