# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

import natsort

from archai.common.file_utils import CHECKPOINT_REGEX
from lm_eval.evaluator import make_table
from lm_eval_harness.lm_eval_hf_model import HFEvalModel
from lm_eval_harness.tasks.human_eval import HumanEval
from lm_eval.evaluator import evaluate
from transformers import AutoTokenizer, CodeGenForCausalLM


def find_checkpoints(folder_name: str) -> str:
    folder_content = os.listdir(folder_name)
    
    checkpoints = [
        os.path.join(folder_name, path)
        for path in folder_content
        if CHECKPOINT_REGEX.search(path) is not None
        and os.path.isdir(os.path.join(folder_name, path))
    ]
    checkpoints = natsort.natsorted(checkpoints)

    return checkpoints


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    for checkpoint in find_checkpoints("codegen-hf"):
        model = CodeGenForCausalLM.from_pretrained(checkpoint)
        hf_model = HFEvalModel(model, tokenizer)

        results = evaluate(
            lm=hf_model,
            task_dict={
                "human_eval": HumanEval(n_samples=1, temperature=0.01, pass_at_k=[1])
            },
        )
            
        output_json = json.dumps(results, indent=2)
        with open(os.path.join(checkpoint, "human_eval.json"), "w") as f:
            f.write(output_json)

        print(make_table(results))
