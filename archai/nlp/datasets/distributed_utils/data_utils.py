# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import logging
import os
import re

import sacremoses
import torch

from archai.nlp.datasets import distributed_utils
from archai.nlp.datasets.corpus import Corpus

def get_lm_corpus(datadir:str, cachedir:str, dataset:str, vocab_type:str,
                  vocab_size:Optional[int]=None, refresh_cache=False):
    corpus = Corpus(datadir, dataset, vocab_type, cachedir,
                    vocab_size=vocab_size, refresh_cache=refresh_cache)
    if not corpus.load(): # if cached version doesn't exist
        corpus.train_and_encode()
        with distributed_utils.distributed.sync_workers() as rank:
            if rank == 0 and not dataset == 'lm1b':
                corpus.save()

    return corpus

def tokenize_raw(text, lang='en'):
    mt = sacremoses.MosesTokenizer(lang)
    text = mt.tokenize(text, return_str=True)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&apos;', "'", text)
    text = re.sub(r'(\d)\.(\d)', r'\1 @.@ \2', text)
    text = re.sub(r'(\d),(\d)', r'\1 @,@ \2', text)
    text = re.sub(r'(\w)-(\w)', r'\1 @-@ \2', text)
    return text


if __name__ == '__main__':
    # test code

    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--cachedir', type=str, default='~/logdir/data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8', 'olx'],
                        help='dataset name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    corpus = get_lm_corpus(args.datadir, args.cachedir, args.dataset, vocab='word')
    logging.info('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
