from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
import contextlib
import os
from typing import Optional

import torch

from archai.nlp.nvidia_transformer_xl.nvidia_utils.vocabulary import Vocab
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed
from archai.nlp.tokenizer_utils.token_trainer import create_tokenizer

# Class GptVocab has been adapted from
# https://github.com/cybertronai/transformer-xl/blob/master/utils/vocabulary.py
class GptVocab(Vocab):
    def __init__(self, max_size=None, vocab_dir:Optional[str]=None):
        from pytorch_transformers import GPT2Tokenizer

        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None

        if vocab_dir is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            tokenizer_files = TokenizerFiles.from_path(save_dir=vocab_dir)
            self.tokenizer = create_tokenizer(tokenizer_files, TokenConfig())

        self.EOT = self.tokenizer.encoder['<|endoftext|>']
        self.max_size = max_size

        pad = 8
        vocab_size = len(self.tokenizer)
        padded_vocab_size = (vocab_size + pad - 1) // pad * pad
        for i in range(0, padded_vocab_size - vocab_size):
            token = f'madeupword{i:09d}'
            self.tokenizer.add_tokens([token])

    def __len__(self):
        return len(self.tokenizer)

    def count_file(self, path, verbose=False, add_eos=False):
        raise RuntimeError('count_file should not be called in GptVocab')

    def build_vocab(self):
        pass

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False) -> torch.LongTensor:
        # Suppress warnings about length.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            out = torch.LongTensor(self.tokenizer.encode(f.read()) + [self.EOT])
            return out

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        return self.tokenizer.encode(line)

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(symbols)

