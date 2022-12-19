# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/pdf/1905.07830.pdf
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from overrides import overrides

from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import HarnessCall, call_factory


class HellaSwagHarnessTask(HarnessTask):
    """HellaSwag harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "hellaswag",
            dataset_config_name="default",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="accuracy",
            metric_config_name=None,
        )

    @property
    def has_test_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return sample["query"]

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        return f" {sample['choices'][sample['label']]}"

    def _get_preprocessed_text(self, text: str) -> str:
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")

        return text

    @overrides
    def _pre_process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        context = f"{sample['ctx_a']} {sample['ctx_b'].capitalize()}"

        return {
            "query": self._get_preprocessed_text(sample["activity_label"] + ": " + context),
            "choices": [self._get_preprocessed_text(ending) for ending in sample["endings"]],
            "label": int(sample["label"]),
        }

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        return [call_factory.log_likelihood(context, f" {choice}") for choice in sample["choices"]]

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = np.argmax(results)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)
