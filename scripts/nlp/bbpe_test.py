from typing import Optional, Union, List
import logging
import argparse
import os

from archai.common import utils, common

from archai.nlp.tokenizer_utils.bbpe_vocab import BbpeVocab
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils

def basic_test():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='PyTorch Transformer-XL Language Model')
    parser.add_argument('--work_dir', default='~/logdir', type=str,
                         help='Directory for the results')
    parser.add_argument('--data', type=str, default=None,
                         help='Location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'wt2', 'lm1b', 'enwik8', 'text8', 'olx'],
                         help='Dataset name')
    parser.add_argument('--vocab_size', type=int, default=10000,
                         help='Size of vocabulary')
    args, _ = parser.parse_known_args()

    pt_data_dir, pt_output_dir = common.pt_dirs()
    args.work_dir = utils.full_path(pt_output_dir or args.work_dir, create=True)
    args.data = args.data or pt_data_dir or common.default_dataroot()
    args.data = utils.full_path(os.path.join(args.data, 'textpred', exp_utils.dataset_dir_name(args.dataset)))

    save_path = utils.full_path(os.path.join(args.work_dir, 'textpred_tests',
                                             'bpe_test', str(args.vocab_size), 'add_prefix_space'),
                                create=True)

    train_filepath = os.path.join(args.data, 'wiki.test.tokens')

    vocab = BbpeVocab(save_path, args.vocab_size, eos_token='<eos>', bos_token='<bos>', unk_token='<unk>')
    vocab.train([train_filepath])

    print('tokenizer len', len(vocab))
    print('BOS', vocab.token_to_id('<bos>'))

    enc = vocab.encode_text("It's a nice sunny day; she murmered. Should we get take-out? ")
    print(enc)
    print(vocab.decode_text(enc))
    print()

    enc = vocab.encode_text("\n It's a nice sunny day; she murmered. Should we get take-out?")
    print(enc)
    print(vocab.decode_text(enc))
    print()

    enc = vocab.encode_text("<bos>\n It's a nice sunny day; she murmered. Should we get take-out?<eos>")
    print(enc)
    print(vocab.decode_text(enc))
    print()

    enc = vocab.encode_text("<bos>\n It's a nice sunny day; she murmered. Should we get take-out?<eos>")
    print(enc)
    print(vocab.decode_text(enc, skip_special_tokens=True))
    print()

if __name__ == "__main__":
    basic_test()