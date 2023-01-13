# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from archai.nlp.datasets.hf.tokenizer_utils.token_config import TokenConfig
from archai.nlp.datasets.hf.tokenizer_utils.tokenizer_base import TokenizerBase


@pytest.fixture
def token_config():
    return TokenConfig(
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        sep_token="<SEP>",
        pad_token="<PAD>",
        cls_token="<CLS>",
        mask_token="<MASK>",
    )


@pytest.fixture
def tokenizer():
    return Tokenizer(BPE())


@pytest.fixture
def trainer():
    return BpeTrainer()


def test_tokenizer_base(token_config, tokenizer, trainer):
    # Assert that the tokenizer base is initialized correctly
    tokenizer_base = TokenizerBase(token_config, tokenizer, trainer)
    assert isinstance(tokenizer_base, TokenizerBase)

    # Assert that the tokenizer can be saved
    tokenizer_base.save("tokenizer.json")
    assert os.path.exists("tokenizer.json")
    assert os.path.exists("token_config.json")

    os.remove("tokenizer.json")
    os.remove("token_config.json")
