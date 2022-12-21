# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories
https://arxiv.org/pdf/1604.01696.pdf
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from overrides import overrides

from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import HarnessCall, call_factory


class StoryCloze2016HarnessTask(HarnessTask):
    """StoryCloze 2016 harness task."""

    def __init__(
        self,
        dataset_config_name: Optional[str] = "2016",
        dataset_dir: Optional[str] = None,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "story_cloze",
            dataset_config_name=dataset_config_name,
            dataset_dir=dataset_dir,
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="accuracy",
            metric_config_name=None,
        )

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return " ".join(
            [
                sample["input_sentence_1"],
                sample["input_sentence_2"],
                sample["input_sentence_3"],
                sample["input_sentence_4"],
            ]
        )

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        labels = [sample["sentence_quiz1"], sample["sentence_quiz2"]]
        return " " + labels[sample["answer_right_ending"] - 1]

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        choices = [sample["sentence_quiz1"], sample["sentence_quiz2"]]
        lls = [call_factory.log_likelihood(context, f" {choice}") for choice in choices]

        return lls

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = np.argmax(results)
        reference = sample["answer_right_ending"] - 1

        self.metric.add(predictions=prediction, reference=reference)


class StoryCloze2018HarnessTask(StoryCloze2016HarnessTask):
    """StoryCloze 2018 harness task."""

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset_config_name="2018",
            dataset_dir=dataset_dir,
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
        )
