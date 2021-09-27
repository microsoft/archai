
from collections import OrderedDict
from typing import Counter, List, Optional
import logging
import os
import json

from overrides import overrides

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from archai.nlp.nvidia_transformer_xl import nvidia_utils as nv_utils
from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.token_config import TokenConfig
from archai.common import utils, common
from archai.nlp.tokenizer_utils.special_token_enum import SpecialTokenEnum

class BbpeVocab(VocabBase):
    def __init__(self, save_path:str, vocab_size:int, pad_vocab_size=False,
                 bos_token:Optional[str]="_BOS_", eos_token:Optional[str]=None,
                 unk_token:Optional[str]="_OOV_", pad_token:Optional[str]=None,
                 min_frequency:Optional[int]=None, model_max_length:Optional[int]=None,
                 add_prefix_space=True,add_prefix_new_line=False, sorted_vocab=True) -> None:
        self._config = TokenConfig(bos_token=bos_token, eos_token=eos_token,
                                   unk_token=unk_token, pad_token=pad_token,
                                   add_prefix_space=add_prefix_space, add_prefix_new_line=add_prefix_new_line)
        self._tokenizer:Optional[PreTrainedTokenizerFast] = None # will load existing or create new
        self._tokenizer_filepath = os.path.join(utils.full_path(save_path, create=True), 'bbpe_tokenizer.json')
        self.vocab_size = vocab_size
        self.sorted_vocab = sorted_vocab
        self.min_frequency = min_frequency
        self.model_max_length = model_max_length

        self.bos_id = []
        self.eos_id = []

        self.pad_vocab_size = pad_vocab_size # make vocab size multiple of 8
        self.pad = 8
        self.padded_vocab_size = self.vocab_size if not self.pad_vocab_size else (self.vocab_size + self.pad - 1) // self.pad * self.pad

    @overrides
    def train(self, filepaths: List[str]) -> None:
        with nv_utils.distributed.sync_workers() as rank:
            if rank == 0:
                logging.info(f'Training BBPE Vocab for size {self.vocab_size} at "{self._tokenizer_filepath}" ...')
                self._train_tokenizer(filepaths)

                if self.sorted_vocab:
                    self.load()
                    self._rewrite_json_sorted(filepaths)
        self.load()

    @overrides
    def special_token_id(self, sp:SpecialTokenEnum)->Optional[int]:
        return self.token_to_id(self._config.special_token_name(sp))

    @overrides
    def token_to_id(self, t:str)->int:
        return self._tokenizer.convert_tokens_to_ids(t)

    @overrides
    def id_to_token(self, id:int)->str:
        return self._tokenizer.convert_ids_to_tokens(id)

    @overrides
    def load(self)->None:
        # TODO: below shouldn't be required: https://github.com/huggingface/transformers/issues/664
        #tokenizer.padding_side = "left"

        # TODO: check is loaded tokenizer has same settings as in vocab constructor
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=self._tokenizer_filepath,
            model_max_length=self.model_max_length,
            bos_token=self._config.bos_token, eos_token = self._config.eos_token,
            unk_token=self._config.unk_token, pad_token = self._config.pad_token)

        self._finalize_tokenizer()

        # these IDs will be used to manually add BOS and EOS
        self.bos_id = [] if not self._config.bos_token else [self.token_to_id(self._config.bos_token)]
        self.eos_id = [] if not self._config.eos_token else [self.token_to_id(self._config.eos_token)]

        logging.info(f'tokenizer len: {len(self._tokenizer)}')
        logging.info(f'tokenizer filepath: {self._tokenizer_filepath}')

    def _rewrite_json_sorted(self, filepaths:List[str]):

        tokens_counter = self._count_token_freq(filepaths)

        logging.info('Saving sorted vocab file...')
        tokens_counter.update(list(range(len(self._tokenizer)))) # add 1 to each value, to ensure that all of them > 0
        min_sort_id = 256 + len(self._config.get_special_tokens())
        sorted_ids = list(range(min_sort_id)) + \
                            [int(token_id) for token_id, _ in tokens_counter.most_common() if int(token_id) >= min_sort_id]

        t_map = [(new, old) for new, old in enumerate(sorted_ids)]
        t_map.sort(key=lambda t:t[1])
        orig2sorted_ids = [t[0] for t in t_map]

        with open(self._tokenizer_filepath, encoding="utf-8") as f:
            tok_json = json.load(f)
        vocab_orig = tok_json['model']['vocab']

        assert len(vocab_orig) == len(orig2sorted_ids)
        v_map = OrderedDict([(vocab, orig2sorted_ids[idx]) for vocab, idx in vocab_orig.items()])

        # save unsorted file
        utils.copy_file(self._tokenizer_filepath, self._tokenizer_filepath + '.unsorted.json')

        tok_json['model']['vocab'] = v_map
        with open(self._tokenizer_filepath, 'w', encoding="utf-8") as f:
            f.write(json.dumps(tok_json, ensure_ascii=False, indent=2))

    def _finalize_tokenizer(self):
        # TODO: EOT is supposed to be added at the end of the file but currently its not done
        # self.EOT = self.tokenizer.bos_token_id # .encoder['<|endoftext|>']

        # TODO: measure impact of padding and remove additional complexity
        if self.pad_vocab_size:
            vocab_size = len(self._tokenizer)
            self.padded_vocab_size = (vocab_size + self.pad - 1) // self.pad * self.pad
            for i in range(0, self.padded_vocab_size - vocab_size):
                token = f'madeupword{i:09d}'
                self._tokenizer.add_tokens([token])

    @overrides
    def is_trained(self)->bool:
        # if any files exist in the directory
        return os.path.isfile(self._tokenizer_filepath)

    @overrides
    def encode_text(self, text:str, add_special_tokens=False)->List[int]:
        text = self._preprocess_text(text)

        # we always set add_special_tokens=False because Huggingface implementation is buggy
        # instead add bos and eos manually
        # https://github.com/huggingface/transformers/issues/3311
        toks = self._tokenizer.encode(text, add_special_tokens=False)

        if add_special_tokens:
            toks = self.bos_id + toks + self.eos_id

        return toks

    @overrides
    def decode_text(self, ids:List[int],skip_special_tokens=False)->str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @overrides
    def __len__(self):
        return self._tokenizer.get_vocab_size()

def _encode_line(line:str, token_config:TokenConfig, tokenizer:ByteLevelBPETokenizer,
                bos_id:List[int], eos_id:List[int])->List[int]:
    line = line.strip()
    if not line:
        return []
    if token_config.add_prefix_space:
        line = ' ' + line
    if token_config.add_prefix_new_line:
        line = '\n' + line
    return bos_id + tokenizer.encode(line).ids + eos_id

def _count_token_freq(filepaths:List[str], tokenizer:ByteLevelBPETokenizer, token_config:TokenConfig,
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
            toks = _encode_line(l, token_config, tokenizer, bos_id, eos_id)
            tokens_counter.update(toks)

    return tokens_counter

def _save_sort_tokens(tokens_counter:Counter, tokenizer:ByteLevelBPETokenizer,
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

def _train_tokenizer(filepaths:List[str], token_config: TokenConfig,
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

def _create_tokenizer(tokenizer_files:TokenizerFiles, token_config: TokenConfig, max_length=1024)->ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_files.vocab_file, tokenizer_files.merges_file)

    # TODO: below shouldn't be required: https://github.com/huggingface/transformers/issues/664
    #tokenizer.padding_side = "left"
    return tokenizer
