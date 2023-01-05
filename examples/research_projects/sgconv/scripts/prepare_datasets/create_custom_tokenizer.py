# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from codexs.data import load_dataset, merge_datasets
from codexs.models import (
    BERTTokenizer,
    BLOOMTokenizer,
    CharTokenizer,
    CodeGenTokenizer,
    CTRLTokenizer,
    GPT2Tokenizer,
    GPTTokenizer,
    LongformerTokenizer,
    OPTTokenizer,
    ReformerTokenizer,
    RobertaTokenizer,
    RoFormerTokenizer,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLNetTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a custom tokenizer")

    parser.add_argument(
        "-t",
        "--tokenizer_type",
        type=str,
        choices=[
            "bert",
            "bloom",
            "char",
            "codegen",
            "ctrl",
            "gpt",
            "gpt2",
            "longformer",
            "opt",
            "reformer",
            "roberta",
            "roformer",
            "transfo_xl",
            "xlm",
            "xlnet",
        ],
        default="transfo_xl",
        help="Type of the tokenizer to use",
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        nargs="+",
        type=str,
        default="wikitext",
        help="Name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-dcn",
        "--dataset_config_name",
        nargs="+",
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
    args.dataset_name = [args.dataset_name] if not isinstance(args.dataset_name, list) else args.dataset_name
    args.dataset_config_name = (
        [args.dataset_config_name] if not isinstance(args.dataset_config_name, list) else args.dataset_config_name
    )

    available_tokenizers = {
        "bert": BERTTokenizer(),
        "bloom": BLOOMTokenizer(),
        "char": CharTokenizer(),
        "codegen": CodeGenTokenizer(),
        "ctrl": CTRLTokenizer(),
        "gpt": GPTTokenizer(),
        "gpt2": GPT2Tokenizer(),
        "longformer": LongformerTokenizer(),
        "opt": OPTTokenizer(),
        "reformer": ReformerTokenizer(),
        "roberta": RobertaTokenizer(),
        "roformer": RoFormerTokenizer(),
        "transfo_xl": TransfoXLTokenizer(),
        "xlm": XLMTokenizer(),
        "xlnet": XLNetTokenizer(),
    }

    tokenizer = available_tokenizers[args.tokenizer_type]

    dataset_list = [
        load_dataset(dn, dcn, dataset_split="train") for dn, dcn in zip(args.dataset_name, args.dataset_config_name)
    ]
    dataset = merge_datasets(dataset_list)
    assert "train" in dataset.keys(), "`train` split must be available when tokenizing a new dataset"

    tokenizer.train_from_iterator(dataset["train"])
    tokenizer.save(args.output_file)
