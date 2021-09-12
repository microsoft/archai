
from typing import List, Optional
import logging
import os

from overrides import overrides

from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase, GPT2TokenizerFast, GPT2Tokenizer, PreTrainedTokenizer
from tokenizers import ByteLevelBPETokenizer

from archai.nlp.nvidia_transformer_xl import nvidia_utils as nv_utils
from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.tokenizer_files import TokenizerFiles
from archai.nlp.tokenizer_utils.token_config import TokenConfig
from archai.nlp.tokenizer_utils.token_trainer import train_tokenizer, create_tokenizer
from archai.common import utils, common

class Gpt2Vocab(VocabBase):
    def __init__(self, vocab_size:int, save_path:str, max_length=1024, pad_vocab_size=True,
                 bos_token:Optional[str]="<|endoftext|>", eos_token:Optional[str]="<|endoftext|>",
                 unk_token:Optional[str]="<|endoftext|>", pad_token:Optional[str]=None,
                 add_prefix_space=False, add_prefix_new_line=False) -> None:
        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None
        # default vocab size for GPT-2 is 50257

        self._config = TokenConfig(bos_token=bos_token, eos_token=eos_token,
                                   unk_token=unk_token, pad_token=pad_token,
                                   add_prefix_space=add_prefix_space, add_prefix_new_line=add_prefix_new_line)
        self._files = TokenizerFiles.from_path(save_path)
        self._tokenizer:Optional[PreTrainedTokenizerFast] = None
        self.save_path = utils.full_path(save_path, create=True) if save_path else save_path

        self.pad_vocab_size = pad_vocab_size
        self.vocab_size = vocab_size
        self.max_length = max_length

    @overrides
    def train(self, filepaths: List[str]) -> None:
        with nv_utils.distributed.sync_workers() as rank:
            if rank == 0:
                logging.info(f'Training GPT2 Vocab for size {self.vocab_size} at "{self.save_path}" ...')
                train_tokenizer(filepaths, self._config, vocab_size=self.vocab_size, save_dir=self.save_path)

        self.load()

    @overrides
    def load(self)->None:
        self._tokenizer = create_tokenizer(self._files, self._config, max_length=self.max_length)
        if self.pad_vocab_size:
            self._finalize_tokenizer()

        logging.info(f'tokenizer len: {len(self._tokenizer)}')
        logging.info(f'merges_file: {self._files.merges_file}')
        logging.info(f'vocab_file: {self._files.vocab_file}')

    def _finalize_tokenizer(self):
        # TODO: EOT is supposed to be added at the end of the file but currently its not done
        # self.EOT = self.tokenizer.bos_token_id # .encoder['<|endoftext|>']

        pad = 8
        vocab_size = len(self._tokenizer)
        padded_vocab_size = (vocab_size + pad - 1) // pad * pad
        for i in range(0, padded_vocab_size - vocab_size):
            token = f'madeupword{i:09d}'
            self._tokenizer.add_tokens([token])

    @overrides
    def is_trained(self)->bool:
        return TokenizerFiles.files_exists(self.save_path)

    @overrides
    def encode_line(self, line)->List[int]:
        return self._tokenizer.encode(line)

    @overrides
    def __len__(self):
        return len(self._tokenizer)

def train_tokenizer(filepaths: List[str], token_config: TokenConfig,
                    vocab_size: int, save_dir: str, save_prefix='tokenizer',
                    dropout: float = None, min_frequency: int = 0, show_progress=False,
                    added_tokens: List[str] = []) -> TokenizerFiles:

    # check if we already have tokenizer cached filed
    tokenizer_out_files = TokenizerFiles.from_path(save_dir=save_dir)
    # if utils.is_debugging() and os.path.exists(tokenizer_out_files.vocab_file) \
    #         and os.path.exists(tokenizer_out_files.merges_file):
    #     logging.info(f'Found GPT2 tokenizer cached files at "{save_dir}", reusing them.')
    #     return tokenizer_out_files

    lines = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines.extend(f.readlines())

    # TODO: measure impact of dropout
    tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=token_config.add_prefix_space)

    # function that will return us batches of lines
    def batch_iterator(batch_size=1000):
        for i in range(0, len(lines), batch_size):
            yield lines[i : i + batch_size]

    special_tokens = token_config.get_special_tokens()

    # train
    tokenizer.train_from_iterator(batch_iterator(),
        vocab_size=vocab_size-len(added_tokens), # original GPT2: 50257
        show_progress=show_progress,
        min_frequency=min_frequency,
        # for GPT2, pad token is not used: https://github.com/huggingface/transformers/issues/2630
        special_tokens=special_tokens)

    # additional tokens we might want to add
    if len(added_tokens):
        tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return tokenizer_out_files

def create_tokenizer(tokenizer_files:TokenizerFiles, token_config: TokenConfig, max_length=1024)->PreTrainedTokenizerFast:
    tokenizer = GPT2TokenizerFast(vocab_file=tokenizer_files.vocab_file,
                              merges_file=tokenizer_files.merges_file,
                              model_max_length=max_length,
                              eos_token=token_config.eos_token,
                              bos_token=token_config.bos_token,
                              unk_token=token_config.unk_token,
                              pad_token=token_config.pad_token)

    # TODO: below shouldn't be required: https://github.com/huggingface/transformers/issues/664
    #tokenizer.padding_side = "left"
    return tokenizer
