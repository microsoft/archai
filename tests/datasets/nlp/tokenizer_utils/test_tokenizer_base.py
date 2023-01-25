# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from overrides import overrides

from archai.datasets.nlp.tokenizer_utils.token_config import SpecialTokenEnum
from archai.datasets.nlp.tokenizer_utils.tokenizer_base import TokenizerBase


@pytest.fixture
def tokenizer_base():
    class Tokenizer(TokenizerBase):
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

    return Tokenizer()


def test_tokenizer_base_len(tokenizer_base):
    assert len(tokenizer_base) == 100


def test_tokenizer_base_train(tokenizer_base):
    tokenizer_base.train(["file1", "file2"])
    assert tokenizer_base.is_trained() is True


def test_tokenizer_base_load(tokenizer_base):
    tokenizer_base.load()
    assert tokenizer_base.is_trained() is True


def test_tokenizer_base_encode_text(tokenizer_base):
    assert tokenizer_base.encode_text("test") == [1, 2, 3]


def test_tokenizer_base_decode_text(tokenizer_base):
    assert tokenizer_base.decode_text([1, 2, 3]) == "decoded"


def test_tokenizer_base_special_token_id(tokenizer_base):
    assert tokenizer_base.special_token_id(SpecialTokenEnum.BOS) == 1
    assert tokenizer_base.special_token_id(SpecialTokenEnum.EOS) == 2
    assert tokenizer_base.special_token_id(SpecialTokenEnum.UNK) == 3
    assert tokenizer_base.special_token_id(SpecialTokenEnum.PAD) == 4
    assert tokenizer_base.special_token_id("invalid") is None


def test_tokenizer_base_token_to_id(tokenizer_base):
    assert tokenizer_base.token_to_id("test") == 5


def test_tokenizer_base_id_to_token(tokenizer_base):
    assert tokenizer_base.id_to_token(5) == "token"
