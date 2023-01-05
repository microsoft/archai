# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Any, Dict

from codexs.core import ArchaiPreTrainedTokenizer, ArchaiPreTrainedTokenizerFast
from codexs.data import load_dataset
from codexs.utils.general_utils import xor
from datasets import DatasetDict, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenizes a dataset.")

    parser.add_argument(
        "output_dataset_file",
        type=str,
        help="Path to the output tokenized dataset file.",
    )

    parser.add_argument(
        "-tc",
        "--token_config_path",
        type=str,
        default=None,
        help="Path to the token's configuration.",
    )

    parser.add_argument(
        "-tk",
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer's configuration.",
    )

    parser.add_argument(
        "-htk",
        "--hub_tokenizer_path",
        type=str,
        default=None,
        help="Name or path to the Hub's tokenizer.",
    )

    parser.add_argument(
        "-ctx",
        "--context_length",
        type=int,
        default=2048,
        help="Length of context (sequence length).",
    )

    parser.add_argument(
        "-sep",
        "--separator",
        type=str,
        default="<|endoftext|>",
        help="Separator string.",
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-dcn",
        "--dataset_config_name",
        type=str,
        default=None,
        help="Configuration name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-ds",
        "--dataset_split",
        type=str,
        default=None,
        help="Splits of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-dk",
        "--dataset_disk",
        type=str,
        default="",
        help="Path of disk-saved dataset.",
    )

    parser.add_argument(
        "-mcn",
        "--mapping_column_name",
        type=str,
        default="code",
        help="Name of column to be mapped during tokenization.",
    )

    parser.add_argument(
        "-np",
        "--n_proc",
        type=int,
        default=32,
        help="Number of processes when mapping dataset.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # Sanity checks
    assert xor(
        args.tokenizer_path, args.hub_tokenizer_path
    ), "`tokenizer_path` and `hub_tokenizer_path` are mutually exclusive."

    if args.tokenizer_path:
        tokenizer = ArchaiPreTrainedTokenizerFast(
            token_config_file=args.token_config_path,
            tokenizer_file=args.tokenizer_path,
        )
    if args.hub_tokenizer_path:
        tokenizer = ArchaiPreTrainedTokenizer.from_pretrained(args.hub_tokenizer_path)

    # `load_dataset` will always return a DatasetDict with a `train` split
    # which will be used to remove the columns
    dataset = load_dataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        dataset_disk=args.dataset_disk,
    )

    def _tokenize(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Concatenates all examples into a single one separated by `separator`
        example = args.separator.join(examples[args.mapping_column_name])

        # Tokenizes the unified example
        tokenized_example = tokenizer(
            [example],
            truncation=False,
            max_length=None,
            return_overflowing_tokens=False,
            return_length=False,
        )

        # Creates a batch of constant-length examples
        seq_length = len(tokenized_example["input_ids"][0])
        batch_input_ids, batch_attention_mask = [], []
        for i in range(0, seq_length, args.context_length):
            input_ids = tokenized_example["input_ids"][0][i : i + args.context_length]
            attention_mask = tokenized_example["attention_mask"][0][i : i + args.context_length]
            assert len(input_ids) == len(attention_mask)

            if len(input_ids) == args.context_length:
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)

        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

    tokenized_dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=args.n_proc,
        batch_size=1000,
    )

    tokenized_dataset.save_to_disk(args.output_dataset_file)
