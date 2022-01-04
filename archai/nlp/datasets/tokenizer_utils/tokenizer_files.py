# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tokenization-related files handling.
"""

from __future__ import annotations

import os
from typing import Optional


class TokenizerFiles:
    """Implements utilities to check whether tokenization files are available.

    """

    def __init__(self, vocab_file: str, merges_file: str) -> None:
        """Overrides initialization method.

        Args:
            vocab_file: Path to the vocabulary file.
            merges_file: Path to the merge file.

        """

        self.vocab_file = vocab_file
        self.merges_file = merges_file

    @staticmethod
    def files_exists(save_dir: str) -> bool:
        """Checks whether tokenization files exists.

        Args:
            save_dir: Folder where files are located.

        Returns:
            (bool): Whether files exists or not.

        """

        files = TokenizerFiles.from_path(save_dir)

        return os.path.exists(files.merges_file) and os.path.exists(files.vocab_file)

    @staticmethod
    def from_path(save_dir: str,
                  save_prefix: Optional[str] = 'tokenizer') -> TokenizerFiles:
        """Gathers tokenization files from specified path.

        Args:
            save_dir: Folder where files are located.
            save_prefix: Prefix used in files.

        Returns:
            (TokenizerFiles): Tokenizer-based files.

        """

        return TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                              merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))
