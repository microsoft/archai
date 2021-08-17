from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
import contextlib
import os
from typing import Optional
from collections import OrderedDict, Counter

import torch

from pytorch_transformers import GPT2Tokenizer

from archai.nlp.nvidia_transformer_xl.nvidia_utils.vocabulary import Vocab
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed
from archai.nlp.tokenizer_utils.token_trainer import create_tokenizer
from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
from archai.nlp.tokenizer_utils.token_trainer import train_tokenizer, create_tokenizer

from tokenizers import ByteLevelBPETokenizer

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
        self.EOT = self.tokenizer.bos_token_id # .encoder['<|endoftext|>']

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
                print('Creating GPT vocab...')
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
        print('Encoding files...')
        with open(path, encoding='utf-8') as f:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                out = torch.LongTensor(self.tokenizer.encode(f.read()) + [self.EOT])
                return out
        print('Encoding files done.')

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        return self.tokenizer.encode(line)

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(symbols)

class BPEVocab(Vocab):
    def __init__(self, max_size:int, vocab_dir:str):
        self.max_size, self.vocab_dir = max_size, vocab_dir
        self.counter = Counter()
        self.min_freq = 2

        # initialize tokenizer
        self.tokenizer = ByteLevelBPETokenizer()
    
    def count_file(self, path, verbose=True, add_eos=False):
        """Setup counter with frequencies, return tokens for the entir file"""
        if verbose:
            print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        # start training
        self.tokenizer.train(files=[path], vocab_size=self.max_size, min_frequency=2, special_tokens=[ "<s>",  "<pad>", "</s>", "<unk>"])

        # read lines, count frequencies of tokens, convert to tokens
        sents = [] # will contain all parsed tokens
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenizer.encode(line.strip()).tokens
                self.counter.update(symbols)
                sents.append(symbols)
        
        # save files
        if not os.path.exists(self.vocab_dir):
            os.makedirs(self.vocab_dir)
        self.tokenizer.save_model(self.vocab_dir)

        return sents
    
    def build_vocab(self):
        """Build the vocab by creating indices from the counter"""
        print('Building vocab with min_freq={}, max_size={}'.format(
            self.min_freq, self.max_size))
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        for sym, cnt in self.counter.most_common(self.max_size):
            self.add_symbol(sym)
        self.add_symbol("<unk>")
        
        self.unk_idx = self.sym2idx['<unk>']

        print('final vocab size is {}, unique tokens are {}'.format(
            len(self), len(self.counter)))
    
    def encode_file(self, path, ordered=False, verbose=True, add_eos=True,
                    add_double_eos=False, model_ext=None):
        if verbose:
            print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenizer.encode(line.strip()).tokens
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)
            return encoded

        return encoded
    
    def tokenize(self, line, add_eos=False, add_double_eos=False):
        """Split line into tokens, add_eos: add special to end, add_double_eos: add special to begin and end"""
        line = line.strip()
        symbols = self.tokenizer.encode(line.strip()).tokens
        return symbols
    
    def convert_to_text(self, indices, vocab_type):
        #print(indices)
        #print(self.tokenizer.decode(indices))
        return self.tokenizer.decode(indices)



