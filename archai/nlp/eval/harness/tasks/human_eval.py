# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from overrides import overrides

from archai.nlp.eval.harness.harness_task import HarnessTask
from archai.nlp.eval.harness.harness_utils import HarnessCall, call_factory


class HumanEvalHarnessTask(HarnessTask):
    """HumanEval harness task."""

    def __init__(
        self,
        dataset_split: Optional[Union[str, List[str]]] = None,
        dataset_cache: Optional[str] = None,
        dataset_samples: Optional[int] = -1,
        random_seed: Optional[int] = 42,
        num_proc: Optional[int] = None,
        n_samples: Optional[int] = 100,
        pass_at_k: Optional[List[int]] = [1, 10, 100],
        temperature: Optional[float] = 0.8,
    ) -> None:
        super().__init__(
            "openai_humaneval",
            dataset_config_name="openai_humaneval",
            dataset_split=dataset_split,
            dataset_cache=dataset_cache,
            dataset_samples=dataset_samples,
            random_seed=random_seed,
            num_proc=num_proc,
            metric_name="code_eval",
            metric_config_name=None,
        )

        self.n_samples = n_samples
        self.pass_at_k = pass_at_k
        self.temperature = temperature

    @overrides
    def _create_inputs(self, sample: Dict[str, Any]) -> str:
        return f"{sample['prompt']}"

    @overrides
    def _create_label(self, sample: Dict[str, Any]) -> str:
        return f"\n{sample['canonical_solution']}"

    @overrides
    def create_sampling_calls(self, sample: Dict[str, Any], context: str) -> Tuple[HarnessCall, ...]:
        return [
            call_factory.generate(
                context,
                stop_tokens=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                max_new_tokens=300,
            )
            for _ in range(self.n_samples)
        ]

    @overrides
    def compute_results(self, sample: Dict[str, Any], results: Tuple[Any, ...]) -> None:
        test_case = sample["test"]
        entry_point = f"check({sample['entry_point']})"

        prediction = [list(results)]
        reference = [f"{test_case}\n{entry_point}"]

        self.metric.add_batch(predictions=prediction, references=reference)

    @overrides
    def compute_metrics(self) -> Dict[str, Any]:
        return self.metric.compute(k=self.pass_at_k)[0]
