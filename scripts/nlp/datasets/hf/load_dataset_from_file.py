# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.nlp.datasets.hf.loaders import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loads a dataset from local files.")

    parser.add_argument(
        "dataset_files",
        nargs="+",
        type=str,
        help="Path to local files that will be used to load dataset.",
    )

    parser.add_argument(
        "-dft",
        "--dataset_files_type",
        type=str,
        choices=[
            "csv",
            "json",
            "parquet",
            "text",
        ],
        default="text",
        help="Type of local files.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(args.dataset_files_type, dataset_files=args.dataset_files)
    print(dataset)
