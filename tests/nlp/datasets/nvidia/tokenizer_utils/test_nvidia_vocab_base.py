# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from overrides import overrides

from archai.nlp.datasets.nvidia.tokenizer_utils.token_config import SpecialTokenEnum
from archai.nlp.datasets.nvidia.tokenizer_utils.vocab_base import VocabBase


@pytest.fixture
def vocab_base():
    class Vocab(VocabBase):
        def __init__(self):
            self.is_trained_value = False

        def __len__(self):
            return 100

        @overrides
        def train(self, filepaths):
            self.is_trained_value = True

        @overrides
        def is_trained(self):
            return self.is_trained_value

        @overrides
        def load(self):
            self.is_trained_value = True

        @overrides
        def encode_text(self, text):
            return [1, 2, 3]

        @overrides
        def decode_text(self, ids):
            return "decoded"

        @overrides
        def special_token_id(self, sp):
            if sp == SpecialTokenEnum.BOS:
                return 1
            if sp == SpecialTokenEnum.EOS:
                return 2
            if sp == SpecialTokenEnum.UNK:
                return 3
            if sp == SpecialTokenEnum.PAD:
                return 4
            return None

        @overrides
        def token_to_id(self, t):
            return 5

        @overrides
        def id_to_token(self, id):
            return "token"

    return Vocab()


def test_vocab_base_len(vocab_base):
    assert len(vocab_base) == 100


def test_vocab_base_train(vocab_base):
    vocab_base.train(["file1", "file2"])
    assert vocab_base.is_trained() is True


def test_vocab_base_load(vocab_base):
    vocab_base.load()
    assert vocab_base.is_trained() is True


def test_vocab_base_encode_text(vocab_base):
    assert vocab_base.encode_text("test") == [1, 2, 3]


def test_vocab_base_decode_text(vocab_base):
    assert vocab_base.decode_text([1, 2, 3]) == "decoded"


def test_vocab_base_special_token_id(vocab_base):
    assert vocab_base.special_token_id(SpecialTokenEnum.BOS) == 1
    assert vocab_base.special_token_id(SpecialTokenEnum.EOS) == 2
    assert vocab_base.special_token_id(SpecialTokenEnum.UNK) == 3
    assert vocab_base.special_token_id(SpecialTokenEnum.PAD) == 4
    assert vocab_base.special_token_id("invalid") is None


def test_vocab_base_token_to_id(vocab_base):
    assert vocab_base.token_to_id("test") == 5


def test_vocab_base_id_to_token(vocab_base):
    assert vocab_base.id_to_token(5) == "token"
