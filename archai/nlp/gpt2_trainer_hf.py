import argparse
from typing import Optional, Tuple, List, Union
import os
import logging
from pytorch_lightning import callbacks

import torch
from torch import nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoConfig
from tokenizers import ByteLevelBPETokenizer
import pytorch_lightning as pl

from archai.common import ml_utils, utils, common
from archai.nlp.token_dataset import TokenDataset, TokenizerFiles, TokenConfig
from archai.nlp.transformer_lightning import TransformerLightning
from archai.nlp.trainer_callback import TrainerCallback


def train_tokenizer(files: Union[str, List[str]], token_config: TokenConfig,
                    vocab_size: int, save_dir: str, save_prefix='tokenizer',
                    dropout: float = None, min_frequency: int = 2,
                    added_tokens: List[str] = []) -> TokenizerFiles:

    assert isinstance(files, str) or isinstance(files, list ), "files must be a string or a list."
    assert isinstance(added_tokens, list), "added_tokens must be a list."

    if isinstance(files, str):
        files = [files]

    tokenizer = ByteLevelBPETokenizer(dropout=dropout)

    tokenizer.train(files=files, vocab_size=vocab_size-len(added_tokens),
        min_frequency=min_frequency,
        special_tokens=[token_config.bos_token, token_config.eos_token, token_config.unk_token])

    tokenizer.add_tokens(added_tokens)

    # generates save_prefix-vocab.json and save_prefix-merges.txt
    tokenizer.save_model(save_dir, save_prefix)

    return TokenizerFiles(vocab_file=os.path.join(save_dir, save_prefix + '-vocab.json'),
                          merges_file=os.path.join(save_dir, save_prefix + '-merges.txt'))

def main():
    parser = argparse.ArgumentParser(description='GPT2 trainer')
    parser.add_argument('--experiment-name', '-n', default='train_gpt2')
    parser.add_argument('--experiment-description', '-d', default='Train GPT2')
    parser.add_argument('--epochs', '-e', type=int, default=108)
    parser.add_argument('--device', default='',
                        help='"cuda" or "cpu" or "" in which case use cuda if available')
    parser.add_argument('--train-batch-size', '-b', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--seed', '-s', type=float, default=42)
    parser.add_argument('--half', type=lambda x: x.lower() == 'true',
                        nargs='?', const=True, default=False)

    parser.add_argument('--datadir', default='',
                        help='where to find dataset files, default is ~/torchvision_data_dir')
    parser.add_argument('--outdir', default='',
                        help='where to put results, default is ~/logdir')

    parser.add_argument('--train-file', default='wiki.train.tokens', # 'tiny_shakespeare.txt'
                        help='training text file')
    parser.add_argument('--vocab-size', type=int, default=5000)
    parser.add_argument('--num-steps', type=int, default=5000 if not utils.is_debugging() else 1)

    args = parser.parse_args()

    pt_data_dir, pt_output_dir = common.pt_dirs()
    if not args.datadir:
        args.datadir = common.default_dataroot()
    if not args.outdir:
        args.outdir = os.environ.get('PT_OUTPUT_DIR', '')
        if not args.outdir:
            args.outdir = os.path.join('~/logdir', 'gpt2_trainer', args.experiment_name)

    expdir = utils.full_path(args.outdir)
    os.makedirs(expdir, exist_ok=True)
    outdir = utils.full_path(args.outdir)
    datadir = pt_data_dir #utils.full_path(args.datadir)

    utils.setup_cuda(args.seed)

    utils.create_logger(filepath=os.path.join(expdir, 'logs.log'))

    # log config for reference
    logging.info(
        f'exp_name="{args.experiment_name}", exp_desc="{args.experiment_description}"')
    logging.info('seed={args.seed}, epochs={args.epochs}, half={args.half}')
    logging.info(f'datadir="{datadir}"')
    logging.info(f'pt_data_dir="{pt_data_dir}"')
    logging.info(f'expdir="{expdir}"')
    logging.info(f'train_batch_size={args.train_batch_size}')

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    token_config = TokenConfig()
    train_file = os.path.join(datadir, 'textpred', 'wikitext-103', args.train_file)
    num_steps = args.num_steps
    vocab_size:int = args.vocab_size
    tokenizer_files = train_tokenizer(files=train_file, token_config=token_config,
                    vocab_size=vocab_size, save_dir=outdir)

    # model, tokenizer = create_models(tokenizer_files, token_config, vocab_size)
    # model.to(device)
    # train_model(train_file, tokenizer_files, token_config, model, outdir,
    #             num_steps=num_steps)

    # print(generate(model, tokenizer, 'I wanted to write you this email because'))

if __name__ == '__main__':
    main()
