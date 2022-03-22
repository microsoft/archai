# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Byte-level Byte-Pair Encoding (BPE) tokenizer.
"""

from typing import Optional
from archai.nlp.datasets_v2.tokenizer_utils.tokenizer_base import ArchaiTokenizer

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


class BBPETokenizer(ArchaiTokenizer):
    """Byte-level Byte-Pair Encoding (BPE) tokenizer, where sequences are tokenized
        according to a byte-level definition, using a Byte-Pair Encoding algorithm.

    """

    def __init__(self,
                 tokenizer_path: Optional[str] = None,
                 token_config_path: Optional[str] = None,
                 min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = 10000,
                 bos_token: Optional[str] = '_BOS_',
                 unk_token: Optional[str] = '_OOV_',
                 pad_token: Optional[str] = '_PAD_',
                 model_max_length: Optional[int] = None,
                 add_prefix_space: Optional[bool] = True,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False,
                 encode_special_tokens: Optional[bool] = False,
                 decode_special_tokens: Optional[bool] = False) -> None:
        """Initializes a BBPE-based tokenizer by setting pre-defined `tokenizer` and `trainer`.

        Args:
            tokenizer_path: Path to the output pre-trained tokenizer file.
            token_config_path: Path to the output token's configuration file.
            min_freq: Minimum frequency of tokens (0 for disabling argument).
            vocab_size: Maximum size of vocabulary.
            bos_token: Begin-of-sentence token.
            unk_token: Unknown token.
            pad_token: Padding token.
            model_max_length: Maximum length of sequences.
            add_prefix_space: Whether a space should be added as a sequence prefix.
            add_prefix_new_line: Whether a new line should be added as a sequence prefix.
            lower_case: Applies lower case to all sequences.
            encode_special_tokens: Whether special tokens should be used to encode sequences.
            decode_special_tokens: Whether special tokens should be used to decode sequences.

        """

        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size,
                             min_frequency=min_freq,
                             special_tokens=[bos_token, unk_token, pad_token])

        super().__init__(tokenizer,
                         trainer,
                         tokenizer_path=tokenizer_path,
                         token_config_path=token_config_path,
                         min_freq=min_freq,
                         vocab_size=vocab_size,
                         bos_token=bos_token,
                         unk_token=unk_token,
                         pad_token=pad_token,
                         model_max_length=model_max_length,
                         add_prefix_space=add_prefix_space,
                         add_prefix_new_line=add_prefix_new_line,
                         lower_case=lower_case,
                         encode_special_tokens=encode_special_tokens,
                         decode_special_tokens=decode_special_tokens)
