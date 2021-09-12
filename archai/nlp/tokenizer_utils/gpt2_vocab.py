
from typing import List, Optional
import logging

from overrides import overrides

from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase, GPT2TokenizerFast, GPT2Tokenizer, PreTrainedTokenizer

from archai.nlp.nvidia_transformer_xl import nvidia_utils as nv_utils
from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.tokenizer_files import TokenizerFiles
from archai.nlp.tokenizer_utils.token_config import TokenConfig
from archai.nlp.tokenizer_utils.token_trainer import train_tokenizer, create_tokenizer

class Gpt2Vocab(VocabBase):
    def __init__(self, vocab_size:int, save_path:str, max_length=1024, pad_vocab_size=True,
                 bos_token:Optional[str]="<|endoftext|>", eos_token:Optional[str]="<|endoftext|>",
                 unk_token:Optional[str]="<|endoftext|>", pad_token:Optional[str]=None, add_prefix_space=False) -> None:
        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None

        self._config = TokenConfig(bos_token=bos_token, eos_token=eos_token,
                                   unk_token=unk_token, pad_token=pad_token,
                                   add_prefix_space=add_prefix_space)
        self._files = TokenizerFiles.from_path(save_path)
        self._tokenizer:Optional[PreTrainedTokenizerFast] = None
        self.save_path = save_path
        self.pad_vocab_size = pad_vocab_size
        self.vocab_size = vocab_size
        self.max_length = max_length

    @overrides
    def train(self, filepaths: List[str]) -> None:
        with nv_utils.distributed.sync_workers() as rank:
            if rank == 0:
                token_config = TokenConfig()
                logging.info(f'Training BBPE Vocab for size {self.vocab_size} at "{self.save_path}" ...')
                lines = []
                for filepath in filepaths:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines.extend(f.readlines())

                train_tokenizer(lines, token_config, vocab_size=self.vocab_size, save_dir=self.save_path)

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
        return self._tokenizer.encode(line).ids

    @overrides
    def __len__(self):
        return len(self._tokenizer)