# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Any, Dict

from codexs.core import ArchaiPreTrainedTokenizer
from codexs.data import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenizes the GitHub-Code dataset.")

    parser.add_argument(
        "output_dataset_file",
        type=str,
        help="Path to the output tokenized dataset file.",
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

    context_length = 2048
    languages = {"C++": "//", "GO": "//", "Java": "//", "JavaScript": "//", "Python": "#"}

    tokenizer = ArchaiPreTrainedTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    dataset = load_dataset(
        dataset_name="codeparrot/github-code",
        dataset_config_name="all-all",
    )

    def _tokenize(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Finds allowed examples based on their programming languages
        # and adds prefix that identifies the language
        allowed_examples = []
        for code, language in zip(examples["code"], examples["language"]):
            if language in languages.keys():
                prefix = f"{languages[language]} language: {language}\n"
                allowed_examples.append(prefix + code)

        # Concatenates all examples into a single one separated by `separator`
        joined_example = "<|endoftext|>".join(allowed_examples)

        # Tokenizes the unified example
        tokenized_example = tokenizer(
            [joined_example],
            truncation=False,
            max_length=None,
            return_overflowing_tokens=False,
            return_length=False,
        )

        # Creates a batch of constant-length examples
        seq_length = len(tokenized_example["input_ids"][0])
        batch_input_ids, batch_attention_mask = [], []
        for i in range(0, seq_length, context_length):
            input_ids = tokenized_example["input_ids"][0][i : i + context_length]
            attention_mask = tokenized_example["attention_mask"][0][i : i + context_length]
            assert len(input_ids) == len(attention_mask)

            if len(input_ids) == context_length:
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
