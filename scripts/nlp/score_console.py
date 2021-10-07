import logging

import torch

from archai.nlp.scoring.score import score
from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils

def test_score():
    device = torch.device('cuda')

    data_dir, work_dir, cache_dir, dataroot = exp_utils.get_create_dirs(data_dir=None, dataset_name='wt103', experiment_name='test_score')

    corpus = get_lm_corpus(datadir=data_dir, cachedir=cache_dir, dataset='wt103', vocab_type='word',
                           vocab_size=None, refresh_cache=False)
    ntokens = len(corpus.vocab)
    logging.info(f'Dataset load complete, vocab size is: {ntokens}')

    model = MemTransformerLM(ntokens)
    model.to(device)

    score(model=model, vocab=corpus.vocab, in_filetype='console')

if __name__ == "__main__":
    exp_utils.script_init()



    test_score()
