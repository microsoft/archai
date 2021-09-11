from typing import Optional, Union, List
import argparse
import os

from archai.common import utils, common

from archai.nlp.tokenizer_utils.token_trainer import train_tokenizer, create_tokenizer
from archai.nlp.tokenizer_utils.tokenizer_files import TokenizerFiles
from archai.nlp.tokenizer_utils.token_config import TokenConfig
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils

def basic_test():
    parser = argparse.ArgumentParser(description='PyTorch Transformer-XL Language Model')
    parser.add_argument('--work_dir', default='~/logdir', type=str,
                         help='Directory for the results')
    parser.add_argument('--data', type=str, default=None,
                         help='Location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'wt2', 'lm1b', 'enwik8', 'text8'],
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
    with open(train_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    token_config = TokenConfig(bos_token='<bos>', eos_token='<eos>', unk_token='_unk_', pad_token='pad', add_prefix_space=True)

    tokenizer_files = train_tokenizer(lines, token_config,
        vocab_size=args.vocab_size, save_dir=save_path)

    tokenizer = create_tokenizer(tokenizer_files, token_config)

    print('tokenizer len', len(tokenizer))
    print('merges_file', tokenizer_files.merges_file)
    print('vocab_file', tokenizer_files.vocab_file)

    enc = tokenizer.encode("It's a nice sunny day; she murmered. Should we get take-out? ")
    print(enc)
    print()

    enc = tokenizer.encode("\n It's a nice sunny day; she murmered. Should we get take-out?")
    print(enc)
    print()

    enc = tokenizer.encode("<bos>\n It's a nice sunny day; she murmered. Should we get take-out?<eos>")
    print(enc)
    print()

if __name__ == "__main__":
    basic_test()