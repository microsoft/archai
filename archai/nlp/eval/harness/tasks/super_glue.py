# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
https://w4ngatang.github.io/static/papers/superglue.pdf
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from overrides import overrides

from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import (
    HarnessCall,
    call_factory,
    clean_sample_text,
)


class AXbHarnessTask(HarnessTask):
    """AX-b harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="axb",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="axb",
        )

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


class AXgHarnessTask(HarnessTask):
    """AX-g harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="axg",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="axg",
        )

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['premise']}\nQuestion: {sample['hypothesis']} True or False?\nAnswer:"

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


class BoolQHarnessTask(HarnessTask):
    """BoolQ harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="boolq",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="boolq",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer:"

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


class CBHarnessTask(HarnessTask):
    """CB harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="cb",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="cb",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['premise']}\nQuestion: {sample['hypothesis']}. True, False or Neither?\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        available_labels = {0: "True", 1: "False", 2: "Neither"}
        label = sample["label"]

        return f" {available_labels[label]}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        ll_true = call_factory.log_likelihood(context, " True")
        ll_false = call_factory.log_likelihood(context, " False")
        ll_neither = call_factory.log_likelihood(context, " Neither")

        return ll_true, ll_false, ll_neither

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = np.argmax(results)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class COPAHarnessTask(HarnessTask):
    """COPA harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="copa",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="copa",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    def _get_text_from_choice(self, choice: str) -> str:
        return choice[0].lower() + choice[1:]

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        connector = {"cause": "because", "effect": "therefore"}

        question = connector[sample["question"]]
        premise = sample["premise"].strip()[:-1]

        return f"{premise} {question}"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        choice = sample["choice1"] if sample["label"] == 0 else sample["choice2"]

        return f" {self._get_text_from_choice(choice)}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        choice1 = f" {self._get_text_from_choice(sample['choice1'])}"
        choice2 = f" {self._get_text_from_choice(sample['choice2'])}"

        ll_choice1 = call_factory.log_likelihood(context, choice1)
        ll_choice2 = call_factory.log_likelihood(context, choice2)

        return ll_choice1, ll_choice2

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = np.argmax(results)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class MultiRCHarnessTask(HarnessTask):
    """MultiRC harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="multirc",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="multirc",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    def _get_answer_from_sample(self, answer: str, label: str) -> str:
        text_label = "yes" if label else "no"

        return f" {answer}\nIs the answer correct? {text_label}"

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['paragraph']}\nQuestion: {sample['question']}\nAnswer:"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        return self._get_answer_from_sample(sample["answer"], sample["label"])

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        false = self._get_answer_from_sample(sample["answer"], False)
        true = self._get_answer_from_sample(sample["answer"], True)

        ll_false = call_factory.log_likelihood(context, false)
        ll_true = call_factory.log_likelihood(context, true)

        return ll_false, ll_true

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        ll_false, ll_true = results

        prediction = {"idx": sample["idx"], "prediction": int(ll_true > ll_false)}
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class ReCoRDHarnessTask(HarnessTask):
    """ReCoRD harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="record",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="record",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    def _get_answer_from_sample(self, query: str, entity: str) -> str:
        return f"  - {query}".replace("@placeholder", entity)

    @overrides
    def _pre_process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "passage": sample["passage"],
            "query": sample["query"],
            "entities": sorted(list(set(sample["entities"]))),
            "answers": sorted(list(set(sample["answers"]))),
        }

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        passage, *highlights = sample["passage"].strip().split("\n@highlight\n")

        text = passage + "\n\n"
        for highlight in highlights:
            text += f"  - {highlight}.\n"

        return text

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        return self._get_answer_from_sample(sample["query"], sample["answers"][0])

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        return [
            call_factory.log_likelihood(context, self._get_answer_from_sample(sample["query"], entity))
            for entity in sample["entities"]
        ]

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        pred_idx = np.argmax(np.array([result for result in results]))

        prediction = {"idx": sample["idx"], "prediction_text": sample["entities"][pred_idx]}
        reference = {"idx": sample["idx"], "answers": sample["answers"]}

        self.metric.add(predictions=prediction, reference=reference)


class WiCHarnessTask(HarnessTask):
    """WiC harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="wic",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="wic",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        word = sample["sentence1"][sample["start1"] : sample["end1"]]
        return f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\nQuestion: Is the word '{word}' used in the same way in the two sentences above?\nAnswer:"

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


class WSCHarnessTask(HarnessTask):
    """WSC harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "super_glue",
            dataset_config_name="wsc",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="super_glue",
            metric_config_name="wsc",
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        raw_passage = sample["text"]
        pre_passage = " ".join(raw_passage.split()[: sample["span2_index"]])
        post_passage = raw_passage[len(pre_passage) + len(sample["span2_text"]) + 1 :]

        passage = clean_sample_text(f"{pre_passage} *{sample['span2_text']}*{post_passage}")
        noun = sample["span1_text"]
        pronoun = sample["span2_text"]

        return f'Passage: {passage}\nQuestion: In the passage above, does the pronoun "{pronoun}" refer to "{noun}"?\nAnswer:'

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
