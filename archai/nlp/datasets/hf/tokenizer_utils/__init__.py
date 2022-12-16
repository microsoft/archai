# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's-related tokenizers, pre-trained tokenizers and token's configuration."""

from archai.nlp.datasets.hf.tokenizer_utils.bert_tokenizer import BertTokenizer
from archai.nlp.datasets.hf.tokenizer_utils.char_tokenizer import CharTokenizer
from archai.nlp.datasets.hf.tokenizer_utils.codegen_tokenizer import CodeGenTokenizer
from archai.nlp.datasets.hf.tokenizer_utils.gpt2_tokenizer import GPT2Tokenizer
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.datasets.hf.tokenizer_utils.transfo_xl_tokenizer import (
    TransfoXLTokenizer,
)
