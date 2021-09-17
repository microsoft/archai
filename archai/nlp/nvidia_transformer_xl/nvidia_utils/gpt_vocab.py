from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
import contextlib
import os
import json
from typing import Optional
from collections import OrderedDict, Counter

import torch, gc

from pytorch_transformers import GPT2Tokenizer

from archai.nlp.nvidia_transformer_xl.nvidia_utils.vocabulary import Vocab
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed
from archai.nlp.tokenizer_utils.token_trainer import create_tokenizer
from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
from archai.nlp.tokenizer_utils.token_trainer import train_tokenizer, create_tokenizer

from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

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
        self.min_freq = 1000
        self.add_prefix_space = True
        self.add_prefix_new_line = True
        self.special_tokens = "_OOV_,_BOS_"
        self.bos_token_id = None
        self.min_sort_id = 256+2

        # initialize tokenizer
        self.tokenizer = ByteLevelBPETokenizer(add_prefix_space=self.add_prefix_space)
    
    def count_file(self, path, verbose=True, add_eos=False):
        """Setup counter with frequencies, return tokens for the entir file"""
        if verbose:
            print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        #if os.path.isfile(os.path.join(self.vocab_dir, "vocab.json.log")):
        #    print('reusing existing vocab {} ...'.format(self.vocab_dir))
        #    self.tokenizer = ByteLevelBPETokenizer.from_file(os.path.join(self.vocab_dir, "vocab.json"), os.path.join(self.vocab_dir, "merges.txt"))
        #    return

        # start training
        self.tokenizer.train(files=path, vocab_size=self.max_size, min_frequency=self.min_freq, special_tokens=self.special_tokens.split(','))

        '''
        # read lines, count frequencies of tokens, convert to tokens
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                #symbols = self.tokenizer.encode(line.strip()).tokens
                line = line.strip()
                if self.add_prefix_space:
                    line = ' ' + line
                if self.add_prefix_new_line:
                    line = '\n' + line
                symbols = self.tokenizer.encode(line).tokens
                self.counter.update(symbols)
        '''

        # save files
        if not os.path.exists(self.vocab_dir):
            try:
                os.makedirs(self.vocab_dir)
            except OSError as error:
                print(error)  
        self.tokenizer.save_model(self.vocab_dir)
        print('saving tokenizer output at {}'.format(self.vocab_dir))

        tokens, tokens_counter = self.tokenize_lines(path)

        # Sort the vocab
        #tokens_counter.update(list(range(self.max_size))) # add 1 to each value, to ensure that all of them > 0
        tokens_counter.update(list(range(len(tokens_counter)))) # add 1 to each value, to ensure that all of them > 0
        print(tokens_counter.most_common(10))
        sorted_ids = list(range(self.min_sort_id)) + \
                     [int(token_id) for token_id, _ in tokens_counter.most_common() if int(token_id) >= self.min_sort_id]

        #orig2sorted_ids = [None] * self.max_size
        orig2sorted_ids = [None] * len(tokens_counter)
        for new, old in enumerate(sorted_ids):
            orig2sorted_ids[old] = new
        
        with open(os.path.join(self.vocab_dir, "vocab.json"), encoding="utf-8") as f:
            vocab_orig = json.load(f)
        
        #vocab_list = [None] * self.max_size
        vocab_list = [None] * len(tokens_counter)
        for vocab, idx in vocab_orig.items():
            vocab_list[orig2sorted_ids[idx]] = vocab
        
        vocab_new = OrderedDict([(v, idx) for idx, v in enumerate(vocab_list)])
        with open(os.path.join(self.vocab_dir, "vocab.json"), 'w', encoding="utf-8") as f:
            f.write(json.dumps(vocab_new, ensure_ascii=False))
        
        self.tokenizer = ByteLevelBPETokenizer.from_file(os.path.join(self.vocab_dir, "vocab.json"), os.path.join(self.vocab_dir, "merges.txt"))
        del tokens
        del tokens_counter
        gc.collect()
        tokens, tokens_counter = self.tokenize_lines(path)

        with open(os.path.join(self.vocab_dir, "vocab.json.log"), 'w', encoding='utf-8') as f:
            #for idx in range(self.max_size):
            for idx in range(len(tokens_counter)):
                f.write(f"{idx}\t{vocab_list[idx]}\t{tokens_counter.get(idx, 0)}\n")
        del tokens
        del tokens_counter
        gc.collect()
    
    def tokenize_lines(self, train_file):
        tokens = []
        tokens_counter = Counter()
        for line in open(train_file):
            line = line.strip()
            if self.add_prefix_space:
                line = ' ' + line

            if self.add_prefix_new_line:
                line = '\n' + line

            if self.bos_token_id is None:
                toks = self.tokenizer.encode(line).ids
            else:
                toks = [self.bos_token_id] + self.tokenizer.encode(line).ids

            tokens.append(toks)
            tokens_counter.update(toks)

        return (tokens, tokens_counter)
    
    def build_vocab(self):
        """Build the vocab by creating indices from the counter"""
        print('Building vocab with min_freq={}, max_size={}'.format(
            self.min_freq, self.max_size))
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        '''
        for sym, cnt in self.counter.most_common(self.max_size):
            self.add_symbol(sym)
        self.add_symbol("<unk>")
        self.unk_idx = self.sym2idx['<unk>']
        '''
        tokenizer_vocab = self.tokenizer.get_vocab()
        self.idx2sym = [0] * len(tokenizer_vocab)
        for bpe_token in tokenizer_vocab:
            self.sym2idx[bpe_token] = tokenizer_vocab[bpe_token]
            self.idx2sym[tokenizer_vocab[bpe_token]] = bpe_token
        self.unk_idx = self.sym2idx['_OOV_']

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
                line = line.strip()
                if self.add_prefix_space:
                    line = ' ' + line
                if self.add_prefix_new_line:
                    line = '\n' + line
                symbols = self.tokenizer.encode(line).tokens
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
        return self.tokenizer.decode(indices)
    
    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))


class Office_PreTokBPEVocab(Vocab):
    def __init__(self, max_size:int, vocab_dir:str):
        self.max_size, self.vocab_dir = max_size, vocab_dir
        self.counter = Counter()
        self.vocab_f = os.path.join(vocab_dir, "vocab.json")
        self.merges_f = os.path.join(vocab_dir, "merges.txt")
        print('loading ByteLevelBPETokenizer from {}'.format(self.vocab_f))
        self.tokenizer = ByteLevelBPETokenizer.from_file(self.vocab_f, self.merges_f)
    
    def count_file(self, path, verbose=True, add_eos=False):
        """Setup counter with frequencies, return tokens for the entir file"""
        # skipping this step

    def build_vocab(self):
        """Build the vocab by creating indices from the counter"""
        print('Reading pretokenized vocab from {}'.format(
            self.vocab_f))
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(self.vocab_f, encoding="utf-8") as f:
            vocab_orig = json.load(f)
        self.idx2sym = [0] * len(vocab_orig)
        for bpe_token, idx in vocab_orig.items():
            self.sym2idx[bpe_token] = idx
            self.idx2sym[idx] = bpe_token
        self.unk_idx = self.sym2idx['_OOV_']

        print('final vocab size is {}'.format(
            len(self.idx2sym)))
    
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
                #encoded.append(torch.LongTensor([int(item) for item in line.strip().split()]))
                encoded += [int(item) for item in line.strip().split()]
        print("creating the gigantic 1d long tensor of size %d"%len(encoded))
        encoded = torch.LongTensor(encoded)
        #if ordered:
        #    encoded = torch.cat(encoded)
        #    return encoded

        return encoded
    
    def encode_file_stream(self, path, ordered=False, verbose=True, add_eos=True,
                    add_double_eos=False, model_ext=None, split_size=100000000):
        if verbose:
            print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                #encoded.append(torch.LongTensor([int(item) for item in line.strip().split()]))
                encoded += [int(item) for item in line.strip().split()]
                if len(encoded) > split_size:
                    yield torch.LongTensor(encoded)
                    encoded = []
        if len(encoded) != 0:
            yield torch.LongTensor(encoded)
        del encoded

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        """Split line into tokens, add_eos: add special to end, add_double_eos: add special to begin and end"""
        line = line.strip()
        print(line)
        symbols = self.tokenizer.encode(line.strip()).tokens
        return symbols
    
    def convert_to_text(self, indices, vocab_type):
        return self.tokenizer.decode(indices)
    

