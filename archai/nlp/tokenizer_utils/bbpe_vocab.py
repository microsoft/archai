
from collections import OrderedDict
from typing import Counter, List, Optional
import logging
import os
import json

from overrides import overrides

from tokenizers import ByteLevelBPETokenizer

from archai.nlp.nvidia_transformer_xl import nvidia_utils as nv_utils
from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.tokenizer_files import TokenizerFiles
from archai.nlp.tokenizer_utils.token_config import TokenConfig
from archai.common import utils, common

class BbpeVocab(VocabBase):
    def __init__(self, save_path:str, vocab_size:int, pad_vocab_size=False,
                 bos_token:Optional[str]="_BOS_", eos_token:Optional[str]=None,
                 unk_token:Optional[str]="_OOV_", min_frequency:Optional[int]=None,
                 add_prefix_space=True,add_prefix_new_line=False, sorted_vocab=True) -> None:
        self._config = TokenConfig(bos_token=bos_token, eos_token=eos_token,
                                   unk_token=unk_token, pad_token=None,
                                   add_prefix_space=add_prefix_space, add_prefix_new_line=add_prefix_new_line)
        self._files = TokenizerFiles.from_path(save_path)
        self._tokenizer:Optional[ByteLevelBPETokenizer] = None # will load existing or create new
        self.save_path = utils.full_path(save_path, create=True) if save_path else save_path
        self.vocab_size = vocab_size
        self.sorted_vocab = sorted_vocab
        self.min_frequency = min_frequency
        self.bos_id = []
        self.eos_id = []

        self.pad_vocab_size = pad_vocab_size # make vocab size multiple of 8
        self.pad = 8
        self.padded_vocab_size = self.vocab_size if not self.pad_vocab_size else (self.vocab_size + self.pad - 1) // self.pad * self.pad

    @overrides
    def train(self, filepaths: List[str]) -> None:
        with nv_utils.distributed.sync_workers() as rank:
            if rank == 0:
                logging.info(f'Training BBPE Vocab for size {self.vocab_size} at "{self.save_path}" ...')
                train_tokenizer(filepaths, self._config, vocab_size=self.vocab_size, save_dir=self.save_path,
                    min_frequency=self.min_frequency if self.min_frequency is not None else 2) # 2 is Huggingface default

                if self.sorted_vocab:
                    self.load()
                    counter = count_token_freq(filepaths, self._tokenizer, self._config, self.bos_id, self.eos_id)
                    save_sort_tokens(counter, self._tokenizer, self._config, self._files)
        self.load()

    @overrides
    def token_to_id(self, t:str)->int:
        return self._tokenizer.token_to_id(t)

    @overrides
    def id_to_token(self, id:int)->str:
        return self._tokenizer.id_to_token(id)

    @overrides
    def load(self)->None:
        self._tokenizer = create_tokenizer(self._files, self._config)
        self._finalize_tokenizer()

        self.bos_id = [] if not self._config.bos_token else [self._tokenizer.token_to_id(self._config.bos_token)]
        self.eos_id = [] if not self._config.eos_token else [self._tokenizer.token_to_id(self._config.eos_token)]

        logging.info(f'tokenizer len: {self._tokenizer.get_vocab_size()}')
        logging.info(f'merges_file: {self._files.merges_file}')
        logging.info(f'vocab_file: {self._files.vocab_file}')

    def _finalize_tokenizer(self):
        # TODO: EOT is supposed to be added at the end of the file but currently its not done
        # self.EOT = self.tokenizer.bos_token_id # .encoder['<|endoftext|>']
        if self.pad_vocab_size:
            vocab_size = self._tokenizer.get_vocab_size()
            self.padded_vocab_size = (vocab_size + self.pad - 1) // self.pad * self.pad
            for i in range(0, padded_vocab_size - vocab_size):
                token = f'madeupword{i:09d}'
                self._tokenizer.add_tokens([token])

    @overrides
    def is_trained(self)->bool:
        return TokenizerFiles.files_exists(self.save_path)

    @overrides
    def encode_line(self, line)->List[int]:
        line = line.strip()
        if self._config.add_prefix_space:
            line = ' ' + line
        if self._config.add_prefix_new_line:
            line = '\n' + line
        return self.bos_id + self._tokenizer.encode(line).ids + self.eos_id

    @overrides
    def decode_line(self, ids:List[int])->str:
        return self._tokenizer.decode(ids)

    @overrides
    def __len__(self):
        return self._tokenizer.get_vocab_size()

def encode_line(line:str, token_config:TokenConfig, tokenizer:ByteLevelBPETokenizer,
                bos_id:List[int], eos_id:List[int])->List[int]:
    line = line.strip()
    if not line:
        return []
    if token_config.add_prefix_space:
        line = ' ' + line
    if token_config.add_prefix_new_line:
        line = '\n' + line
    return bos_id + tokenizer.encode(line).ids + eos_id

def count_token_freq(filepaths:List[str], tokenizer:ByteLevelBPETokenizer, token_config:TokenConfig,
                     bos_id:List[int], eos_id:List[int])->Counter:
    logging.info('Counting token frequencies...')
    tokens_counter = Counter()
    tokens_counter.update(list(range(tokenizer.get_vocab_size()))) # add each token

    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i,l in enumerate(lines):
            if ((i+1)%100000)==0:
                logging.info(f'Counted tokens for line {i+1}...')
            toks = encode_line(l, token_config, tokenizer, bos_id, eos_id)
            tokens_counter.update(toks)

    return tokens_counter

def save_sort_tokens(tokens_counter:Counter, tokenizer:ByteLevelBPETokenizer,
                    token_config:TokenConfig, tokenizer_files:TokenizerFiles):
    logging.info('Saving sorted vocab file...')
    tokens_counter.update(list(range(tokenizer.get_vocab_size()))) # add 1 to each value, to ensure that all of them > 0
    min_sort_id = 256 + len(token_config.get_special_tokens())
    sorted_ids = list(range(min_sort_id)) + \
                        [int(token_id) for token_id, _ in tokens_counter.most_common() if int(token_id) >= min_sort_id]

    t_map = [(new, old) for new, old in enumerate(sorted_ids)]
    t_map.sort(key=lambda t:t[1])
    orig2sorted_ids = [t[0] for t in t_map]

    with open(tokenizer_files.vocab_file, encoding="utf-8") as f:
        vocab_orig = json.load(f)

    assert len(vocab_orig) == len(orig2sorted_ids)
    v_map = OrderedDict([(vocab, orig2sorted_ids[idx]) for vocab, idx in vocab_orig.items()])

    utils.copy_file(tokenizer_files.vocab_file, tokenizer_files.vocab_file + '.unsorted.json')

    with open(tokenizer_files.vocab_file, 'w', encoding="utf-8") as f:
        f.write(json.dumps(v_map, ensure_ascii=False))

def train_tokenizer(filepaths:List[str], token_config: TokenConfig,
                    vocab_size: int, save_dir: str, save_prefix='tokenizer',
                    dropout: float = None, min_frequency: int = 2, show_progress=False,
                    added_tokens: List[str] = []) -> TokenizerFiles:
    logging.info('Training tokenizer...')
    # check if we already have tokenizer cached filed
    tokenizer_out_files = TokenizerFiles.from_path(save_dir=save_dir)
    # if utils.is_debugging() and os.path.exists(tokenizer_out_files.vocab_file) \
    #         and os.path.exists(tokenizer_out_files.merges_file):
    #     logging.info(f'Found BBPE tokenizer cached files at "{save_dir}", reusing them.')
    #     return tokenizer_out_files

    special_tokens = token_config.get_special_tokens()

    # TODO: measure impact of dropout
    tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=token_config.add_prefix_space)

    tokenizer.train(files=filepaths, vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=special_tokens)

    # additional tokens we might want to add
    if len(added_tokens):
        tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return tokenizer_out_files

def create_tokenizer(tokenizer_files:TokenizerFiles, token_config: TokenConfig, max_length=1024)->ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_files.vocab_file, tokenizer_files.merges_file)

    # TODO: below shouldn't be required: https://github.com/huggingface/transformers/issues/664
    #tokenizer.padding_side = "left"
    return tokenizer