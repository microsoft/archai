from typing import Optional, Union, List
import os
import argparse

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase, GPT2TokenizerFast

from archai.common import utils, common
from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils

def train_tokenizer(lines:List[str], token_config: TokenConfig,
                    vocab_size: int, save_dir: str, save_prefix='tokenizer',
                    dropout: float = None, min_frequency: int = 0, show_progress=False,
                    added_tokens: List[str] = []) -> TokenizerFiles:

    tokenizer_out_files = TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                            merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))
    if utils.is_debugging() and os.path.exists(tokenizer_out_files.vocab_file) \
            and os.path.exists(tokenizer_out_files.merges_file):
        return tokenizer_out_files

    # TODO: remove dropout
    tokenizer = ByteLevelBPETokenizer(dropout=dropout, add_prefix_space=token_config.add_prefix_space)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(lines), batch_size):
            yield lines[i : i + batch_size]

    tokenizer.train_from_iterator(batch_iterator(),
        vocab_size=vocab_size-len(added_tokens), # original GPT2: 50257
        show_progress=show_progress,
        min_frequency=min_frequency,
        # for GPT2, pad token is not used: https://github.com/huggingface/transformers/issues/2630
        special_tokens=[token_config.bos_token, token_config.eos_token, token_config.unk_token])

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

def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer-XL Language Model')
    parser.add_argument('--data', type=str, default=None,
                         help='Location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'wt2', 'lm1b', 'enwik8', 'text8'],
                         help='Dataset name')
    parser.add_argument('--vocab', type=str, default='word', choices=['word', 'bpe'],
                         help='Type of vocabulary')
    parser.add_argument('--vocab_size', type=int, default=5000,
                         help='Size of vocabulary')
    args, _ = parser.parse_known_args()

    pt_data_dir, pt_output_dir = common.pt_dirs()
    args.data = args.data or pt_data_dir or common.default_dataroot()
    args.data = utils.full_path(os.path.join(args.data, 'textpred', exp_utils.dataset_dir_name(args.dataset)))

    save_path = os.path.join(args.data,'textpred', 'wikitext-103-bpe-vocab')

    train_filepath = os.path.join(args.data, 'wiki.train.tokens')
    with open(train_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    token_config = TokenConfig()
    tokenizer_files = train_tokenizer(lines, token_config,
        vocab_size=args.vocab_size, save_dir=save_path)
    tokenizer = create_tokenizer(tokenizer_files, token_config)
    print(len(tokenizer))
