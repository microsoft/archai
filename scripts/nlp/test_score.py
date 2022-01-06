import logging
import argparse
import os
from typing import Optional

import torch

from archai.nlp.scoring import score
from archai.nlp.models.mem_transformer import MemTransformerLM
from archai.nlp.datasets.distributed_utils.data_utils import get_lm_corpus
from archai.nlp.datasets import exp_utils
from archai.common import utils, common
from archai.nlp.tokenizer_utils.vocab_base import VocabBase

def create_vocab(dataset_dir:str, dataset_name:str)->VocabBase:
    corpus = get_lm_corpus(datadir=dataset_dir, cachedir=cache_dir, dataset=dataset_name, vocab_type='word', vocab_size=None, refresh_cache=False)

    return corpus.vocab

def test_score(model:Optional[MemTransformerLM], dataset_dir:str, dataset_name:str, in_filetype:str,
               in_filepath:str, out_filepath:str, score_output_dir:str):
    vocab = create_vocab(dataset_dir, dataset_name)
    logging.info(f'Dataset load complete, vocab size is: {len(vocab)}')

    model = MemTransformerLM(len(vocab)) if model is None else model

    score.score(model=model, vocab=vocab, in_filetype=in_filetype,
          in_filepath=in_filepath, out_filepath=out_filepath, score_output_dir=score_out_dir,
          min_score=0.0, min_pred_len=0)

if __name__ == "__main__":
    exp_utils.script_init()

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='PyTorch Transformer-XL Language Model')
    parser.add_argument('--output_dir', default='~/logdir', type=str,
                         help='Directory for the results')
    parser.add_argument('--dataroot', type=str, default=None,
                         help='Location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'wt2', 'lm1b', 'enwik8', 'text8', 'olx'],
                         help='Dataset name')
    parser.add_argument('--in_filetype', type=str, default='console', # console, text, smartcompose
                         help='input for scoring')
    parser.add_argument('--experiment_name', type=str, default='test_score',
                         help='dir name for exoeriment')
    parser.add_argument('--model_filepath', type=str, default='~/dataroot/textpred/experiment_results/baseline_g8/train_xxl_dgx1_8gpu_fp16/checkpoint_best.pt',
                         help='Checkpoint file path for model')
    args, _ = parser.parse_known_args()

    dataset_dir, work_dir, cache_dir, dataroot = exp_utils.get_create_dirs(dataroot=args.dataroot, dataset_name=args.dataset, experiment_name=args.experiment_name, output_dir=args.output_dir)

    eval_filepath = os.path.join(dataroot, 'textpred', 'eval', 'GSuiteCompete10pc_toy.ljson')
    out_filepath = os.path.join(work_dir, 'score_preds.txt')
    score_out_dir = utils.full_path(os.path.join(work_dir, 'scores'), create=True)
    model_filepath = utils.full_path(args.model_filepath)

    print('eval_filepath', eval_filepath)
    print('out_filepath', out_filepath)
    print('score_out_dir', score_out_dir)
    print('model_filepath', model_filepath)


    model, *_ = MemTransformerLM.load_model(model_filepath, model=None, on_cpu=False)

    test_score(model, dataset_dir, args.dataset, args.in_filetype,
               eval_filepath, out_filepath, score_out_dir)
