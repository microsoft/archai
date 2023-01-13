# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM

from archai.nlp.eval.profiler import profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profiles models using eval/profiler package.")

    parser.add_argument("model_path", type=str, help="Path to the model file.")

    parser.add_argument(
        "-n",
        "--n_warmups",
        type=int,
        default=1,
        help="Number of warmups before profiling.",
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

    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    inputs = {"input_ids": torch.zeros((1, model.config.max_length), dtype=torch.long).to(model.device)}
    output["profiler"] = profile(model, model_kwargs=inputs, n_warmups=args.n_warmups)
    output["model"] = model.config.to_dict()

    output_path = os.path.join(
        args.output_dir,
        args.prefix + "profile.json",
    )
    with open(output_path, "w") as f:
        json.dump(output, f)
