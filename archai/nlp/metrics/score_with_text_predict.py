# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Score models using Text Predict.
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
import re
import time

import numpy as np
import pandas as pd
from archai.nlp.metrics.text_predict.prediction import TextPredictionSequence
from archai.nlp.metrics.text_predict.predictor import score


def parse_args():
    parser = argparse.ArgumentParser(description='Score models with Text Predict.')

    try:
        save_path = os.environ['AMLT_OUTPUT_DIR']
    except:
        save_path = '~/logdir' 

    score = parser.add_argument_group('Score configuration')
    score.add_argument('--default_path',
                        type=str,
                        default=save_path,
                        help='Path to the default folder used to save outputs.')

    score.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                        help='Type of model to be searched.')
                    
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    amlt_data = os.environ.get('AMLT_DATA_DIR', '')
    amlt_output = os.environ.get('AMLT_OUTPUT_DIR', '')

    if args.amulet_data:
        args.input = osp.join(amlt_data, args.input)
        
        if args.model_type == 'gpt2onnxprob':
            args.tokenizer = osp.join(amlt_data, args.tokenizer)
        else:
            args.tokenizer = osp.join(osp.dirname(amlt_output), args.model)
            args.tokenizer = osp.dirname(args.tokenizer)

    if args.amulet_model:
        args.model = osp.join(osp.dirname(amlt_output), args.model)

    if args.amulet_output:
        args.output = osp.join(amlt_output, args.output)

    print(f'input: {args.input}')
    print(f'tokenizer: {args.tokenizer}')
    print(f'model: {args.model}')

    args.output = f'{args.input}.pred' if args.output is None else args.output
    args.score_output_dir = f'{args.output}.dir' if args.score_output_dir is None else args.score_output_dir

    print(f'output: {args.output}')

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)

    score(model, vocab, in_filetype, in_filepath, out_filepath, score_output_dir,
          save_step, min_score, max_score, score_step, expected_match_rate,
          current_paragraph_only, do_scoring, max_body_len, min_pred_len)
