# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os
import shutil
from archai.datasets.nlp.tokenizer_utils.gpt2_tokenizer import Gpt2Tokenizer
from archai.datasets.nlp.nvidia_data_loader_utils import LMOrderedIterator, LMMultiFileIterator


def test_lm_ordered_iterator():
    # Assert that iterator returns the correct variables
    input_ids = torch.zeros(256)
    iterator = iter(LMOrderedIterator(input_ids, 1, 8))
    input_ids, labels, seq_len, warmup = next(iterator)
    assert input_ids.shape == (1, 8)
    assert labels.shape == (1, 8)
    assert seq_len == 8
    assert warmup is True

    # Assert that iterator is able to return data with different
    # batch size and sequence length
    input_ids = torch.zeros(512)
    iterator = iter(LMOrderedIterator(input_ids, 2, 16))
    input_ids, labels, seq_len, warmup = next(iterator)
    assert input_ids.shape == (2, 16)
    assert labels.shape == (2, 16)
    assert seq_len == 16
    assert warmup is True


def test_lm_multi_file_iterator():
    input_files = [f"tmp_{i}.txt" for i in range(5)]
    for input_file in input_files:
        with open(input_file, "w") as f:
            [f.write("lm multi file iterator test file") for i in range(10)]

    vocab = Gpt2Tokenizer("tokenizer")
    vocab.train(input_files)

    # Assert that iterator returns the correct variables
    iterator = iter(LMMultiFileIterator(input_files, vocab, 1, 8, n_chunks=2))
    input_ids, labels, seq_len, warmup = next(iterator)
    assert input_ids.shape == (1, 8)
    assert labels.shape == (1, 8)
    assert seq_len == 8
    assert warmup is True
    
    for input_file in input_files:
        os.remove(input_file)
    shutil.rmtree("tokenizer")
