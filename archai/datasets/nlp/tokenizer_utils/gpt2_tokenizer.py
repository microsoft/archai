# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from archai.datasets.nlp.tokenizer_utils.bbpe_tokenizer import BbpeTokenizer


class Gpt2Tokenizer(BbpeTokenizer):
    """GPT-2 based tokenizer."""

    def __init__(
        self,
        save_path: str,
        vocab_size: Optional[int] = 50257,
        pad_vocab_size: Optional[bool] = True,
        bos_token: Optional[str] = "<|endoftext|>",
        eos_token: Optional[str] = "<|endoftext|>",
        unk_token: Optional[str] = "<|unk|>",
        pad_token: Optional[str] = None,
        min_frequency: Optional[int] = None,
        model_max_length: Optional[int] = 1024,
        add_prefix_space: Optional[bool] = True,
        add_prefix_new_line: Optional[bool] = True,
        sorted_vocab: Optional[bool] = True,
    ) -> None:
        """Define the tokenization pipeline.

        Args:
            save_path: Path to save the tokenizer.
            vocab_size: Maximum size of vocabulary.
            pad_vocab_size: Whether vocabulary size should be padded to a multiple of 8.
            bos_token: Begin-of-sentence token.
            eos_token: End-of-sentence token.
            unk_token: Unknown token.
            pad_token: Padding token.
            min_frequency: Minimum frequency of tokens.
            model_max_length: Maximum length of sequence.
            add_prefix_space: Whether a prefix space token should be added.
            add_prefix_new_line: Whether a prefix new line token should be added.
            sorted_vocab: Whether vocabulary should be sorted.

        """

        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None
        # default vocab size for GPT-2 is 50257

        super().__init__(
            save_path=save_path,
            vocab_size=vocab_size,
            pad_vocab_size=pad_vocab_size,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            min_frequency=min_frequency,
            model_max_length=model_max_length,
            add_prefix_space=add_prefix_space,
            add_prefix_new_line=add_prefix_new_line,
            sorted_vocab=sorted_vocab,
        )
