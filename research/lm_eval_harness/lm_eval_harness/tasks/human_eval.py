# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List, Optional

from datasets.arrow_dataset import Dataset
from datasets.download import DownloadMode
from evaluate import load
from lm_eval.base import Task
from lm_eval.metrics import mean
from lm_eval_harness.utils.request_factory import Request, rf

# Allow code evaluation
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class HumanEval(Task):
    VERSION = 0
    DATASET_PATH = "openai_humaneval"
    DATASET_NAME = "openai_humaneval"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode: Optional[DownloadMode] = None,
        n_samples: Optional[int] = 1,
        temperature: Optional[float] = 0.01,
        pass_at_k: Optional[List[int]] = None,
    ) -> None:
        super().__init__(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)

        self.n_samples = n_samples
        self.temperature = temperature
        self.pass_at_k = pass_at_k or [1]

    def should_decontaminate(self) -> bool:
        return False

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self) -> Dataset:
        return self.dataset["test"]

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        return doc["prompt"]

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        return "\n" + doc["canonical_solution"]

    def construct_requests(self, doc: Dict[str, Any], ctx: str) -> List[Request]:
        return [
            rf.generate(
                ctx,
                ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                True,
                self.temperature,
                0.95,
                300,
            )
            for _ in range(self.n_samples)
        ]

    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        test_case = doc["test"]
        entry_point = f"check({doc['entry_point']})"

        prediction = [list(results)]
        reference = [f"{test_case}\n{entry_point}"]

        metric = load("code_eval")
        metric.add_batch(predictions=prediction, references=reference)
        pass_at_k = metric.compute(k=self.pass_at_k)[0]

        return {k: v for k, v in pass_at_k.items()}

    def aggregation(self) -> Dict[str, Any]:
        return {f"pass@{k}": mean for k in self.pass_at_k}

    def higher_is_better(self) -> Dict[str, Any]:
        return {f"pass@{k}": True for k in self.pass_at_k}
