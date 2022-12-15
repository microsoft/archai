# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from archai.nlp.datasets.hf.tokenizer_utils import ArchaiPreTrainedTokenizerFast
from archai.nlp.eval.harness import HarnessModel, evaluate, load_harness_task
from archai.nlp.eval.profiler import profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates pre-trained models using eval/harness package.")

    parser.add_argument("pre_trained_model_path", type=str, help="Path to the pre-trained model file.")

    parser.add_argument(
        "hub_tokenizer_path",
        type=str,
        help="Name or path to the Hub's tokenizer.",
    )

    parser.add_argument("-t", "--tasks", nargs="+", type=str, default=None, help="Tasks to be evaluated.")

    parser.add_argument(
        "-ns",
        "--n_few_shot_samples",
        type=int,
        default=0,
        help="Number of few-shot samples.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Defines an output folder for the saved outputs.",
    )

    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="",
        help="Prefix to append to name of results file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = {}

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Sanity checks
    if args.tasks is None:
        raise ValueError("`--tasks` should have at least one task.")
    if args.prefix:
        args.prefix += "-"

    model = AutoModelForCausalLM.from_pretrained(args.pre_trained_model_path)
    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained(args.hub_tokenizer_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    harness_model = HarnessModel(model, tokenizer)

    # Profiles the model
    inputs = {"input_ids": torch.zeros((1, harness_model.max_length), dtype=torch.long).to(harness_model.device)}
    output["profiler"] = profile(harness_model.model, model_kwargs=inputs, n_warmups=1)
    output["model"] = harness_model.model.config.to_dict()

    for task in args.tasks:
        harness_task = load_harness_task(task, random_seed=args.seed)

        output[task] = evaluate(
            harness_model,
            harness_task,
            n_few_shot=args.n_few_shot_samples,
        )

    output_path = os.path.join(
        args.output_dir,
        args.prefix + f"{harness_model.model_name}-{args.n_few_shot_samples}shot.json",
    )
    with open(output_path, "w") as f:
        json.dump(output, f)
