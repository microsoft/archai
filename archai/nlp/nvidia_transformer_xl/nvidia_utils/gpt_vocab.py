from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
import contextlib
import os
from typing import Optional

import torch

from pytorch_transformers import GPT2Tokenizer

from archai.nlp.nvidia_transformer_xl.nvidia_utils.vocabulary import Vocab
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed
from archai.nlp.tokenizer_utils.token_trainer import create_tokenizer
from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
from archai.nlp.tokenizer_utils.token_trainer import train_tokenizer, create_tokenizer

# Class GptVocab has been adapted from
# https://github.com/cybertronai/transformer-xl/blob/master/utils/vocabulary.py
class GptVocab(Vocab):
    def __init__(self, max_size:int, vocab_dir:str):
        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None

        self.max_size, self.vocab_dir = max_size, vocab_dir
        self._filepaths = []

    def _finalize_tokenizer(self):
        self.EOT = self.tokenizer.encoder['<|endoftext|>']

        pad = 8
        vocab_size = len(self.tokenizer)
        padded_vocab_size = (vocab_size + pad - 1) // pad * pad
        for i in range(0, padded_vocab_size - vocab_size):
            token = f'madeupword{i:09d}'
            self.tokenizer.add_tokens([token])

    def __len__(self):
        return len(self.tokenizer)

    def count_file(self, path, verbose=False, add_eos=False):
        self._filepaths.append(path)

    def build_vocab(self):
        if not self.vocab_dir:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            token_config = TokenConfig()
            if not TokenizerFiles.files_exists(self.vocab_dir):
                lines = []
                for filepath in self._filepaths:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines.extend(f.readlines())

                tokenizer_files = train_tokenizer(lines, token_config,
                    vocab_size=self.max_size, save_dir=self.vocab_dir)
            else:
                tokenizer_files = TokenizerFiles.from_path(self.vocab_dir)

            self.tokenizer = create_tokenizer(tokenizer_files, token_config)
            self._finalize_tokenizer()

        print('tokenizer len', len(self.tokenizer))
        print('merges_file', tokenizer_files.merges_file)
        print('vocab_file', tokenizer_files.vocab_file)

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False) -> torch.LongTensor:
        # Suppress warnings about length.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            out = torch.LongTensor(self.tokenizer.encode(f.read()) + [self.EOT])
            return out

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        return self.tokenizer.encode(line)

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(symbols)

