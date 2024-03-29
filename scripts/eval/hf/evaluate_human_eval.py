# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os

import natsort
from lm_eval.evaluator import evaluate
from lm_eval_harness.lm_eval_hf_model import HFEvalModel
from lm_eval_harness.tasks.human_eval import HumanEval
from transformers import AutoTokenizer, CodeGenForCausalLM

from archai.common.file_utils import CHECKPOINT_REGEX


def find_checkpoints(folder_name: str) -> str:
    folder_content = os.listdir(folder_name)

    checkpoints = [
        os.path.join(folder_name, path)
        for path in folder_content
        if CHECKPOINT_REGEX.search(path) is not None and os.path.isdir(os.path.join(folder_name, path))
    ]
    checkpoints = natsort.natsorted(checkpoints)

    return checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face checkpoints on HumanEval.")

    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory containing the checkpoints to evaluate.",
    )

    parser.add_argument(
        "-htn",
        "--hub_tokenizer_name",
        type=str,
        default="Salesforce/codegen-350M-mono",
        help="Name of the tokenizer to use (via the Hugging Face Hub).",
    )

    parser.add_argument(
        "-ns",
        "--n_samples",
        type=int,
        default=1,
        help="Number of code samples to generate.",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature for the code generation.",
    )

    parser.add_argument(
        "-pk",
        "--pass_at_k",
        type=int,
        nargs="+",
        default=1,
        help="Pass at k for the code generation.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if not isinstance(args.pass_at_k, list):
        args.pass_at_k = [args.pass_at_k]

    tokenizer = AutoTokenizer.from_pretrained(args.hub_tokenizer_name)

    for checkpoint in find_checkpoints(args.checkpoint_dir):
        print(f"Loading checkpoint: {checkpoint}")

        model = CodeGenForCausalLM.from_pretrained(checkpoint)
        hf_model = HFEvalModel(model, tokenizer)

        print("Evaluating on HumanEval ...")
        results = evaluate(
            lm=hf_model,
            task_dict={
                "human_eval": HumanEval(
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                    pass_at_k=args.pass_at_k,
                )
            },
        )

        output_json = json.dumps(results, indent=2)
        output_json_path = os.path.join(checkpoint, "human_eval.json")
        with open(output_json_path, "w") as f:
            print(f"Dumping evaluation results: {output_json_path}")
            f.write(output_json)
