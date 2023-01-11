# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.nlp.datasets.nvidia.corpus import load_corpus
from archai.nlp.datasets.nvidia.corpus_utils import create_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creates a corpus (tokenizer + encoded dataset).")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["wt103"],
        default="wt103",
        help="Name of the dataset.",
    )

    parser.add_argument(
        "-dd",
        "--dataset_dir",
        type=str,
        default="",
        help="Path to the folder that holds the dataset's files.",
    )

    parser.add_argument(
        "-dcd",
        "--dataset_cache_dir",
        type=str,
        default="cache",
        help="Path to the folder that holds the dataset's cache.",
    )

    parser.add_argument(
        "-drc",
        "--dataset_refresh_cache",
        action="store_true",
        help="Whether the dataset's cache should be refreshed.",
    )

    parser.add_argument(
        "-v",
        "--vocab",
        type=str,
        choices=["word", "bbpe", "gpt2"],
        default="gpt2",
        help="Type of vocabulary/tokenizer.",
    )

    parser.add_argument(
        "-vs",
        "--vocab_size",
        type=int,
        default=10000,
        help="Size of the vocabulary.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    args.dataset_dir, _, _, args.dataset_cache_dir = create_dirs(
        args.dataset_dir,
        args.dataset,
        cache_dir=args.dataset_cache_dir,
    )

    dataset = load_corpus(
        args.dataset,
        args.dataset_dir,
        args.dataset_cache_dir,
        args.vocab,
        vocab_size=args.vocab_size,
        refresh_cache=args.dataset_refresh_cache,
    )

    print(dataset)
    print(dataset.vocab)
