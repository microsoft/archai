# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from overrides import overrides

from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import HarnessCall, call_factory


class ARCEasyHarnessTask(HarnessTask):
    """ARC-Easy harness task."""

    def __init__(
        self,
        dataset_config_name: Optional[str] = "ARC-Easy",
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "ai2_arc",
            dataset_config_name=dataset_config_name,
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
        return sample["query"]

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        return f" {sample['choices'][sample['label']]}"

    @overrides
    def _pre_process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        answer_key_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        sample["answerKey"] = answer_key_map.get(sample["answerKey"], sample["answerKey"])

        return {
            "id": sample["id"],
            "query": f"Question: {sample['question']}\nAnswer:",
            "choices": sample["choices"]["text"],
            "label": ["A", "B", "C", "D", "E"].index(sample["answerKey"]),
        }

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        return [call_factory.log_likelihood(context, f" {choice}") for choice in sample["choices"]]

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = np.argmax(results)
        reference = sample["label"]

        self.metric.add(predictions=prediction, reference=reference)


class ARCChallengeHarnessTask(ARCEasyHarnessTask):
    """ARC-Challenge harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset_config_name="ARC-Challenge",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
        )
