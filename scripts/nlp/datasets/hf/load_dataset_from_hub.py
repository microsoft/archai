# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.nlp.datasets.hf.loaders import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loads a dataset from Huggingface's Hub.")

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-dcn",
        "--dataset_config_name",
        type=str,
        default="wikitext-103-raw-v1",
        help="Configuration name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-dsp",
        "--dataset_split",
        type=str,
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="train",
        help="Which split of the dataset to load.",
    )

    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        random_seed=args.seed,
    )
    print(dataset)
