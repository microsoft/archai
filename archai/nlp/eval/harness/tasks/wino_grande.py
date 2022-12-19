# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""WinoGrande: An Adversarial Winograd Schema Challenge at Scale
https://arxiv.org/pdf/1907.10641.pdf
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from overrides import overrides

from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import HarnessCall, call_factory


class WinoGrandeHarnessTask(HarnessTask):
    """WinoGrande harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
    ) -> None:
        super().__init__(
            "winogrande",
            dataset_config_name="winogrande_xl",
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

    def _get_partial_context_from_sample(self, sentence: str, option: str) -> str:
        pronoun_loc = sentence.index("_")
        return sentence[:pronoun_loc] + option

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        answer = sample["answer"]
        option = sample["option" + answer]

        return self._get_partial_context_from_sample(sample["sentence"], option)

    def _get_partial_label_from_sample(self, sentence: str) -> str:
        pronoun_loc = sentence.index("_") + 1
        return f" {sentence[pronoun_loc:].strip()}"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        return self._get_partial_label_from_sample(sample["sentence"])

    def _get_full_context(self, context: str, partial_context: str) -> str:
        context = context.split("\n\n")
        context.pop()

        return "\n\n".join([*context, partial_context]) if context else partial_context

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        label = self._get_partial_label_from_sample(sample["sentence"])
        options = [sample["option1"], sample["option2"]]

        lls = []
        for option in options:
            partial_context = self._get_partial_context_from_sample(sample["sentence"], option)
            context = self._get_full_context(context, partial_context)

            lls.append(call_factory.log_likelihood(context, label))

        return lls

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        prediction = np.argmax(results)
        reference = 0 if sample["answer"] == "1" else 1

        self.metric.add(predictions=prediction, reference=reference)
