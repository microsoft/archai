# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json

from harness.lm_eval_evaluator import evaluate_wrapper
from harness.lm_eval_hf_model import HFEvalModel
from harness.lm_eval_utils import MultiChoice, pattern_match
from lm_eval.evaluator import make_table
from lm_eval.tasks import ALL_TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluates pre-trained models using lm-eval package.")

    parser.add_argument("pre_trained_model_path", type=str, help="Path to the pre-trained model file.")

    parser.add_argument(
        "hub_tokenizer_path",
        type=str,
        help="Name or path to the Hugging Face hub's tokenizer.",
    )

    parser.add_argument(
        "-t",
        "--tasks",
        choices=MultiChoice(ALL_TASKS),
        type=str,
        default=None,
        help="Tasks to be evaluated (separated by comma), e.g., `wsc,cb,copa`.",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Path to the saved outputs.",
    )

    parser.add_argument(
        "-ns",
        "--n_few_shot_samples",
        type=int,
        default=0,
        help="Number of few-shot samples.",
    )

    parser.add_argument(
        "-ls",
        "--limit_samples",
        type=int,
        default=None,
        help="Limit the number of samples.",
    )

    parser.add_argument(
        "-nc",
        "--no_cache",
        action="store_true",
        help="Whether to not store predictions in a cache database.",
    )

    parser.add_argument(
        "-dnp",
        "--decontamination_ngrams_path",
        type=str,
        default=None,
        help="Path to the de-contamination n-grams file.",
    )

    parser.add_argument(
        "-ddp",
        "--description_dict_path",
        type=str,
        default=None,
        help="Path to the description dictionary file.",
    )

    parser.add_argument(
        "-ci",
        "--check_integrity",
        action="store_true",
        help="Whether to check integrity of tasks.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.limit_samples:
        print("Warning: --limit_samples should only be used for testing.")

    task_names = ALL_TASKS if args.tasks is None else pattern_match(args.tasks.split(","), ALL_TASKS)
    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    model = HFEvalModel(args.pre_trained_model_path, args.hub_tokenizer_path)

    outputs = evaluate_wrapper(
        model,
        task_names,
        num_fewshot=args.n_few_shot_samples,
        no_cache=args.no_cache,
        limit=args.limit_samples,
        description_dict=description_dict,
        check_integrity=args.check_integrity,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
    )

    output_json = json.dumps(outputs, indent=2)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(output_json)

    print(make_table(outputs))
