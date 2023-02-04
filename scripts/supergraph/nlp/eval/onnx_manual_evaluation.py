# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from transformers import DataCollatorForLanguageModeling

from archai.nlp.datasets.hf.loaders import encode_dataset, load_dataset
from archai.nlp.datasets.hf.tokenizer_utils import ArchaiPreTrainedTokenizerFast
from archai.nlp.eval.onnx_evaluator import manual_evaluate
from archai.nlp.onnx import load_from_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates an ONNX model.")

    parser.add_argument("onnx_model_path", type=str, help="Path to the pre-trained ONNX model file.")

    parser.add_argument(
        "hub_tokenizer_path",
        type=str,
        help="Name or path to the Hub's tokenizer.",
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained(args.hub_tokenizer_path)
    tokenizer.pad_token_id = 50256

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_dataset = load_dataset(args.dataset_name, args.dataset_config_name, dataset_split="test")
    eval_dataset = encode_dataset(eval_dataset, tokenizer, format_column_name=["input_ids"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    session = load_from_onnx(args.onnx_model_path)

    eval_outputs = manual_evaluate(
        session,
        eval_dataset["test"],
        data_collator=collator,
        batch_size=8,
        n_seed_tokens=2,
        n_accuracy_type=1,
    )
    print(eval_outputs)
