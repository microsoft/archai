# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import torch
import os
import re

import natsort

from lm_eval.evaluator import make_table
from lm_eval_harness.lm_eval_hf_model import HFEvalModel
from lm_eval_harness.tasks.human_eval import HumanEval
from lm_eval.evaluator import evaluate
from transformers import AutoTokenizer, CodeGenConfig, CodeGenForCausalLM

CHECKPOINT_REGEX = re.compile(r"^" + r"(\d+)$")


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

    for checkpoint in find_checkpoints("codegen-ds"):
        state_dict = torch.load(os.path.join(checkpoint, "mp_rank_00_model_states.pt"))
        model_state_dict = state_dict["module"]

        config = CodeGenConfig(
            vocab_size=50295,
            n_positions=2048,
            n_embd=1024,
            n_layer=20, 
            n_head=16,
            rotary_dim=32,
        )
        model = CodeGenForCausalLM(config=config)
        model.load_state_dict(model_state_dict)
        
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
