# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Byte-BPE-based tokenizer."""

import json
import os
from collections import OrderedDict
from typing import Counter, List, Optional, Union

from overrides import overrides
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from archai.common import utils
from archai.nlp import logging_utils
from archai.nlp.datasets.nvidia import distributed_utils
from archai.nlp.datasets.nvidia.tokenizer_utils.token_config import (
    SpecialTokenEnum,
    TokenConfig,
)
from archai.nlp.datasets.nvidia.tokenizer_utils.vocab_base import VocabBase

logger = logging_utils.get_logger(__name__)


class BbpeVocab(VocabBase):
    """Byte-BPE-based vocabulary/tokenizer."""

    def __init__(
        self,
        save_path: str,
        vocab_size: int,
        pad_vocab_size: Optional[bool] = False,
        bos_token: Optional[str] = "_BOS_",
        eos_token: Optional[str] = None,
        unk_token: Optional[str] = "_OOV_",
        pad_token: Optional[str] = None,
        min_frequency: Optional[int] = None,
        model_max_length: Optional[int] = None,
        add_prefix_space: Optional[bool] = True,
        add_prefix_new_line: Optional[bool] = False,
        sorted_vocab: Optional[bool] = True,
        encode_special_tokens: Optional[bool] = False,
        decode_special_tokens: Optional[bool] = False,
    ) -> None:
        """Define the tokenization pipeline.

        Args:
            save_path: Path to save the vocabulary.
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
            encode_special_tokens: Whether special tokens should be encoded.
            decode_special_tokens: Whether special tokens should be decoded.

        """

        self._config = TokenConfig(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_prefix_new_line=add_prefix_new_line,
        )

        self._tokenizer = None
        self._tokenizer_filepath = os.path.join(utils.full_path(save_path, create=True), "bbpe_tokenizer.json")

        self.vocab_size = vocab_size
        self.sorted_vocab = sorted_vocab
        self.min_frequency = min_frequency
        self.model_max_length = model_max_length
        self.encode_special_tokens = encode_special_tokens
        self.decode_special_tokens = decode_special_tokens

        self.bos_id = []
        self.eos_id = []

        self.pad_vocab_size = pad_vocab_size  # vocab_size multiple of 8
        self.pad = 8
        self.padded_vocab_size = (
            self.vocab_size if not self.pad_vocab_size else (self.vocab_size + self.pad - 1) // self.pad * self.pad
        )

    @overrides
    def __len__(self):
        return len(self._tokenizer)

    @overrides
    def train(self, filepaths: List[str]) -> None:
        with distributed_utils.sync_workers() as rank:
            if rank == 0:
                logger.info(f"Training vocabulary with size = {self.vocab_size} at {self._tokenizer_filepath} ...")
                self._train_tokenizer(filepaths)

                if self.sorted_vocab:
                    self.load()
                    self._rewrite_json_sorted(filepaths)

        self.load()

    @overrides
    def is_trained(self) -> bool:
        return os.path.isfile(self._tokenizer_filepath)

    @overrides
    def load(self) -> None:
        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self._tokenizer_filepath,
            model_max_length=self.model_max_length,
            bos_token=self._config.bos_token,
            eos_token=self._config.eos_token,
            unk_token=self._config.unk_token,
            pad_token=self._config.pad_token,
        )
        self._finalize_tokenizer()

        # These IDs will be used to manually add BOS and EOS
        self.bos_id = [] if not self._config.bos_token else [self.token_to_id(self._config.bos_token)]
        self.eos_id = [] if not self._config.eos_token else [self.token_to_id(self._config.eos_token)]

        logger.debug(f"Tokenizer length: {len(self._tokenizer)}")
        logger.debug(f"Tokenizer file path: {self._tokenizer_filepath}")

    @overrides
    def encode_text(self, text: Union[List, str]) -> List[int]:
        
        if isinstance(text, list):
            text = [self._preprocess_text(sentence) for sentence in text]
        else:
            text = self._preprocess_text(text)

        # Always set add_special_tokens=False because Huggingface's implementation is buggy
        # Instead add bos and eos manually
        # https://github.com/huggingface/transformers/issues/3311
        
        if isinstance(text, list):
            toks = self._tokenizer(text, add_special_tokens=False)
        else:
            toks = self._tokenizer.encode(text, add_special_tokens=False)

        if self.encode_special_tokens:
            toks = self.bos_id + toks + self.eos_id

        return toks

    @overrides
    def decode_text(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=self.decode_special_tokens)

    @overrides
    def special_token_id(self, sp: SpecialTokenEnum) -> int:
        return self.token_to_id(self._config.special_token_name(sp))

    @overrides
    def token_to_id(self, t: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(t)

    @overrides
    def id_to_token(self, id: int) -> str:
        return self._tokenizer.convert_ids_to_tokens(id)

    def _rewrite_json_sorted(self, filepaths: List[str]) -> None:
        """Re-write a sorted version of the vocabulary.

        Args:
            filepaths: A list of paths to input files.

        """

        logger.info("Saving sorted vocabulary ...")

        tokens_counter = self._count_token_freq(filepaths)
        # Adds 1 to each value, to ensure that all of them > 0
        tokens_counter.update(list(range(len(self._tokenizer))))

        min_sort_id = 256 + len(self._config.get_special_tokens())
        sorted_ids = list(range(min_sort_id)) + [
            int(token_id) for token_id, _ in tokens_counter.most_common() if int(token_id) >= min_sort_id
        ]

        t_map = [(new, old) for new, old in enumerate(sorted_ids)]
        t_map.sort(key=lambda t: t[1])
        orig2sorted_ids = [t[0] for t in t_map]

        with open(self._tokenizer_filepath, encoding="utf-8") as f:
            tok_json = json.load(f)
        vocab_orig = tok_json["model"]["vocab"]

        assert len(vocab_orig) == len(orig2sorted_ids)
        v_map = OrderedDict([(vocab, orig2sorted_ids[idx]) for vocab, idx in vocab_orig.items()])

        utils.copy_file(self._tokenizer_filepath, self._tokenizer_filepath + ".unsorted.json")
        tok_json["model"]["vocab"] = v_map
        with open(self._tokenizer_filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(tok_json, ensure_ascii=False, indent=2))

    def _finalize_tokenizer(self) -> None:
        """Finalize the tokenizer by padding the vocabulary."""

        if self.pad_vocab_size:
            vocab_size = len(self._tokenizer)

            self.padded_vocab_size = (vocab_size + self.pad - 1) // self.pad * self.pad
            for i in range(0, self.padded_vocab_size - vocab_size):
                token = f"madeupword{i:09d}"
                self._tokenizer.add_tokens([token])

    def _preprocess_text(self, text: str) -> str:
        """Pre-process the text.

        Args:
            text: The input text.

        Returns:
            Pre-processed text.

        """

        text = text.strip()

        # Does not add space for empty lines
        if self._config.add_prefix_new_line and (text == "\n" or text == ""):
            return "\n"
        if self._config.add_prefix_space:
            text = " " + text
        if self._config.add_prefix_new_line:
            text = "\n" + text
        if self._config.lower_case:
            text = text.lower()

        return text

    def _count_token_freq(self, filepaths: List[str]) -> Counter:
        """Count the frequency of tokens.

        Args:
            filepaths: A list of paths to input files.

        Returns:
            Tokens' frequencies.

        """

        logger.info("Counting token frequencies...")

        tokens_counter = Counter()
        tokens_counter.update(list(range(len(self._tokenizer))))

        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, l in enumerate(lines):
                if ((i + 1) % 100000) == 0:
                    logger.info(f"Counted tokens for line {i+1} ...")

                toks = self.encode_text(l)
                tokens_counter.update(toks)

        return tokens_counter

    def _train_tokenizer(
        self, filepaths: List[str], dropout: Optional[float] = None, added_tokens: Optional[List[str]] = None
    ) -> None:
        """Inner loop of tokenizer's training.

        Args:
            filepaths: A list of paths to input files.
            dropout: Dropout ratio.
            added_tokens: Additional tokens.

        """

        logger.info("Training tokenizer ...")

        special_tokens = self._config.get_special_tokens()
        min_frequency = self.min_frequency if self.min_frequency is not None else 2

        # Pre-processes the file for training as well
        def read_line_iter(func, file):
            for line in file:
                yield func(line)
            return

        open_files = [open(filepath, "r") for filepath in filepaths]
        iter_files = iter([read_line_iter(self._preprocess_text, file) for file in open_files])

        # Spaces are added by ourselves
        tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=False)
        tokenizer.train_from_iterator(
            iter_files, vocab_size=self.vocab_size, min_frequency=min_frequency, special_tokens=special_tokens
        )

        for file in open_files:
            file.close()

        if len(added_tokens):
            tokenizer.add_tokens(added_tokens)

        tokenizer.save(self._tokenizer_filepath, pretty=True)
