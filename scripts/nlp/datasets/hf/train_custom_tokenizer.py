# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.nlp.datasets.hf.loaders import load_dataset
from archai.nlp.datasets.hf.tokenizer_utils import (
    BertTokenizer,
    CharTokenizer,
    CodeGenTokenizer,
    GPT2Tokenizer,
    TransfoXLTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a custom tokenizer.")

    parser.add_argument(
        "-t",
        "--tokenizer_type",
        type=str,
        choices=[
            "bert",
            "char",
            "codegen",
            "gpt2",
            "transfo-xl",
        ],
        default="gpt2",
        help="Type of the tokenizer to use.",
    )

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
        "-o",
        "--output_file",
        type=str,
        default="tokenizer.json",
        help="Output file to save the pre-trained tokenizer.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    available_tokenizers = {
        "bert": BertTokenizer(),
        "char": CharTokenizer(),
        "codegen": CodeGenTokenizer(),
        "gpt2": GPT2Tokenizer(),
        "transfo-xl": TransfoXLTokenizer(),
    }

    tokenizer = available_tokenizers[args.tokenizer_type]

    dataset = load_dataset(
        dataset_name=args.dataset_name, dataset_config_name=args.dataset_config_name, dataset_split="train"
    )
    assert "train" in dataset.keys(), "`train` split must be available when tokenizing a new dataset"

    tokenizer.train_from_iterator(dataset["train"])
    tokenizer.save(args.output_file)
