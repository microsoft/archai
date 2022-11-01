# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT2-based tokenizer.
"""

from typing import Optional

from archai.nlp.datasets.nvidia.tokenizer_utils.bbpe_vocab import BbpeVocab


class Gpt2Vocab(BbpeVocab):
    def __init__(self, save_path:str, vocab_size:int=50257, pad_vocab_size=True,
                 bos_token:Optional[str]="<|endoftext|>", eos_token:Optional[str]="<|endoftext|>",
                 unk_token:Optional[str]="<|unk|>", pad_token:Optional[str]=None,
                 min_frequency:Optional[int]=None, model_max_length:Optional[int]=1024,
                 add_prefix_space=True,add_prefix_new_line=True, sorted_vocab=True) -> None:
        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None
        # default vocab size for GPT-2 is 50257

        super().__init__(save_path=save_path, vocab_size=vocab_size, pad_vocab_size=pad_vocab_size,
                         bos_token=bos_token, eos_token=eos_token,
                         unk_token=unk_token, pad_token=pad_token,
                         min_frequency=min_frequency, model_max_length=model_max_length,
                         add_prefix_space=add_prefix_space, add_prefix_new_line=add_prefix_new_line,
                         sorted_vocab=sorted_vocab)
