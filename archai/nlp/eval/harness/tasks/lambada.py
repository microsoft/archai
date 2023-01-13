# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The LAMBADA dataset: Word prediction requiring a broad discourse context
https://arxiv.org/pdf/1606.06031.pdf
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from overrides import overrides

from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import HarnessCall, call_factory


class LambadaHarnessTask(HarnessTask):
    """LAMBADA harness task."""

    def __init__(
        self,
        dataset_name: Optional[str] = "lambada",
        dataset_config_name: Optional[str] = "plain_text",
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset_name,
            dataset_config_name=dataset_config_name,
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="accuracy",
        )

    @property
    def has_train_set(self) -> bool:
        return False

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return sample["text"].rsplit(" ", 1)[0]

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        label = sample["text"].rsplit(" ", 1)[1]

        return f" {label}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        return call_factory.log_likelihood(context, self._create_label(sample), return_exact_match=True)

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        _, is_exact_match = results[0]

        prediction = is_exact_match
        reference = 1

        self.metric.add(predictions=prediction, reference=reference)


class LambadaOpenAIHarnessTask(LambadaHarnessTask):
    """LAMBADA (OpenAI version) harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset_name="craffel/openai_lambada",
            dataset_config_name=None,
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
        )
