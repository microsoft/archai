# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from archai.nlp.datasets.nvidia.tokenizer_utils.token_config import (
    SpecialTokenEnum,
    TokenConfig,
)


@pytest.fixture
def token_config():
    return TokenConfig(
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        add_prefix_space=False,
        add_prefix_new_line=True,
        lower_case=True,
    )


def test_special_token_enum():
    # Assert that the correct values are assigned to the special tokens enumerator
    assert SpecialTokenEnum.UNK.value == 0
    assert SpecialTokenEnum.BOS.value == 1
    assert SpecialTokenEnum.EOS.value == 2
    assert SpecialTokenEnum.PAD.value == 3
    assert SpecialTokenEnum.MASK.value == 4


def test_token_config(token_config):
    # Assert that the correct values are assigned to the special tokens
    assert token_config.bos_token == "<bos>"
    assert token_config.eos_token == "<eos>"
    assert token_config.unk_token == "<unk>"
    assert token_config.pad_token == "<pad>"
    assert token_config.add_prefix_space is False
    assert token_config.add_prefix_new_line is True
    assert token_config.lower_case is True

    # Assert that the special tokens are added to the special token list
    special_tokens = token_config.get_special_tokens()
    assert special_tokens == ["<unk>", "<bos>", "<eos>", "<pad>"]

    # Assert that the special tokens names are returned correctly
    assert token_config.special_token_name(SpecialTokenEnum.BOS) == "<bos>"
    assert token_config.special_token_name(SpecialTokenEnum.EOS) == "<eos>"
    assert token_config.special_token_name(SpecialTokenEnum.UNK) == "<unk>"
    assert token_config.special_token_name(SpecialTokenEnum.PAD) == "<pad>"
    assert token_config.special_token_name("invalid") is None
