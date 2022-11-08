# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datasets import load_dataset

from archai.nlp.datasets.hf.tokenizer_utils.gpt2_tokenizer import GPT2Tokenizer
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)

if __name__ == "__main__":
    pre_trained_tokenizer = ArchaiPreTrainedTokenizerFast(tokenizer_file="a")
    # dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # tokenizer = GPT2Tokenizer()
    # tokenizer.train_from_iterator(dataset["train"])
