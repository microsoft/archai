# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from hashlib import sha1
from typing import List, Optional

import numpy as np
from lm_eval.base import CachingLM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import run_task_tests
from lm_eval_harness.lm_eval_hf_model import HFEvalModel


def evaluate_wrapper(
    hf_model: HFEvalModel,
    tasks: List[str],
    num_fewshot: Optional[int] = 0,
    no_cache: Optional[bool] = False,
    limit: Optional[int] = None,
    bootstrap_iters: Optional[int] = 100000,
    description_dict: Optional[str] = None,
    check_integrity: Optional[bool] = False,
    decontamination_ngrams_path: Optional[str] = None,
):
    random.seed(1234)
    np.random.seed(1234)

    if not no_cache:
        hf_model_id = sha1(repr(hf_model.model).encode("ascii")).hexdigest()
        hf_model = CachingLM(hf_model, f"cache/{hf_model_id}.db")

    if check_integrity:
        run_task_tests(task_list=tasks)

    task_dict = get_task_dict(tasks)
    results = evaluate(
        lm=hf_model,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
    )

    results["config"] = {
        "num_fewshot": num_fewshot,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }

    return results
