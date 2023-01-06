# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List

import pytest

from archai.nlp.datasets.hf.tokenizer_utils.token_config import TokenConfig


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


def test_token_config_special_tokens(token_config):
    # Assert that the special tokens are set correctly
    special_tokens = token_config.special_tokens
    assert isinstance(special_tokens, List)
    assert len(special_tokens) == 7
    assert "<BOS>" in special_tokens
    assert "<EOS>" in special_tokens
    assert "<UNK>" in special_tokens
    assert "<SEP>" in special_tokens
    assert "<PAD>" in special_tokens
    assert "<CLS>" in special_tokens
    assert "<MASK>" in special_tokens


def test_token_config_to_dict(token_config):
    # Assert that the token config is converted to a dictionary correctly
    token_dict = token_config.to_dict()
    assert isinstance(token_dict, Dict)
    assert token_dict["bos_token"] == "<BOS>"
    assert token_dict["eos_token"] == "<EOS>"
    assert token_dict["unk_token"] == "<UNK>"
    assert token_dict["sep_token"] == "<SEP>"
    assert token_dict["pad_token"] == "<PAD>"
    assert token_dict["cls_token"] == "<CLS>"
    assert token_dict["mask_token"] == "<MASK>"


def test_token_config_from_file(token_config, tmp_path):
    token_config_path = tmp_path / "token_config.json"
    token_config.save(str(token_config_path))

    # Assert that the token config is loaded correctly from a file
    loaded_token_config = TokenConfig.from_file(str(token_config_path))
    assert isinstance(loaded_token_config, TokenConfig)
    assert loaded_token_config.bos_token == "<BOS>"
    assert loaded_token_config.eos_token == "<EOS>"
