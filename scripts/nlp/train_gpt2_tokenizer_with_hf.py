# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.datasets.hf_datasets.tokenizer_utils.gpt2_tokenizer import GPT2Tokenizer
from datasets import load_dataset


if __name__ ==  "__main__":
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    tokenizer = GPT2Tokenizer()
    tokenizer.train_from_iterator(dataset["train"])