# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from archai.nlp.datasets.nvidia.lm_iterators import LMOrderedIterator


@pytest.fixture
def iterator():
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    return LMOrderedIterator(input_ids, bsz=1, bptt=5)


def test_lm_ordered_iterator(iterator):
    # Assert that the iterator has the correct number of batches
    assert iterator.n_batch == 4

    # Assert that the batches can be iterated over
    for inputs, labels, seq_len, warmup in iterator:
        assert inputs.shape == (1, 5)
        assert labels.shape == (1, 5)
        assert seq_len == 5
        assert warmup is True
