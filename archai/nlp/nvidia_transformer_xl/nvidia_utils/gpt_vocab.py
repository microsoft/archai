import contextlib
import os
from typing import List, Optional
import logging

import torch

from overrides import overrides

from pytorch_transformers import GPT2Tokenizer

from archai.nlp.nvidia_transformer_xl.nvidia_utils.vocab_base import VocabBase
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed
from archai.nlp.tokenizer_utils.token_trainer import create_tokenizer
from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
from archai.nlp.tokenizer_utils.token_trainer import train_tokenizer, create_tokenizer
from archai.nlp.nvidia_transformer_xl import nvidia_utils as nv_utils
from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles

# Class GptVocab has been adapted from
# https://github.com/cybertronai/transformer-xl/blob/master/utils/vocab.py
class GptVocab(VocabBase):
    def __init__(self, vocab_size:int):
        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None

        self.vocab_size = vocab_size
        self.tokenizer = None # to be created later

    @overrides
    def load(self, path:str)->bool:
        if self.exists(path):
            token_config = TokenConfig()
            tokenizer_files = TokenizerFiles.from_path(path)

            self.tokenizer = create_tokenizer(tokenizer_files, token_config)
            self._finalize_tokenizer()


            logging.info(f'tokenizer len: {len(self.tokenizer)}')
            logging.info(f'merges_file: {tokenizer_files.merges_file}')
            logging.info(f'vocab_file: {tokenizer_files.vocab_file}')

            return True
        else:
            return False

    @overrides
    def exists(self, path)->bool:
        return TokenizerFiles.files_exists(path)

    @overrides
    def train(self, filepaths:List[str], save_dir:str)->None:
        with nv_utils.distributed.sync_workers() as rank:
            if rank == 0:
                token_config = TokenConfig()
                logging.info(f'Training BBPE Vocab for size {self.vocab_size} at "{save_dir}" ...')
                lines = []
                for filepath in filepaths:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines.extend(f.readlines())

                train_tokenizer(lines, token_config, vocab_size=self.vocab_size, save_dir=save_dir)

        self.load(save_dir)

    @overrides
    def encode_line(self, line)->torch.Tensor:
        return torch.LongTensor(self.tokenizer.encode(line).ids)

    def _finalize_tokenizer(self):
        # TODO: EOT is supposed to be added at the end of the file but currently its not done
        # self.EOT = self.tokenizer.bos_token_id # .encoder['<|endoftext|>']

        pad = 8
        vocab_size = len(self.tokenizer)
        padded_vocab_size = (vocab_size + pad - 1) // pad * pad
        for i in range(0, padded_vocab_size - vocab_size):
            token = f'madeupword{i:09d}'
            self.tokenizer.add_tokens([token])

    @overrides
    def __len__(self):
        return len(self.tokenizer)