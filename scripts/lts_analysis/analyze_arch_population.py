# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import copy
import os
import random
import pathlib
import json
from typing import Dict, Tuple, Optional
from collections import namedtuple

from scipy.stats import spearmanr

from archai.common import utils


ParamsPpl = namedtuple('ParamsPpl', ['total_params', 'decoder_params', 'valid_ppl'])


def parse_args():
    parser = argparse.ArgumentParser(description='Analyzes logs of full training of a population of architectures')

    parser.add_argument('--root_folder',
                        type=str,
                        default='~/archaiphilly/amlt/gpt2_flex_random_l5_u15',
                        help='Full path to folder containing training logs')

    parser.add_argument('--out_dir',
                        type=str,
                        default='~/archai_experiment_reports')

    args = parser.parse_args()

    return args


def get_params_and_ppl(train_log_json:str)->Optional[ParamsPpl]:
    with open(train_log_json, 'r') as f:
        lines = f.read().splitlines()
    if not lines:
        return
    else:
        # last line contains n_all_param, n_nonemb_param
        assert lines[-1][:4] == "DLLL"
        data = json.loads(lines[-1][5:])
        total_params = data['data']['n_all_param']
        decoder_params = data['data']['n_nonemb_param']

        # last but one line contains valid perplexity
        assert lines[-2][:4] == "DLLL"
        data = json.loads(lines[-2][5:])
        valid_ppl = data['data']['valid_perplexity']
        params_ppl = ParamsPpl(total_params=total_params, 
                            decoder_params=decoder_params, 
                            valid_ppl=valid_ppl)
        
        return params_ppl



                    
if __name__ == '__main__':
    # Gathers the command line arguments
    args = vars(parse_args())

    # root dir where all results are stored
    results_dir = pathlib.Path(utils.full_path(args['root_folder']))

    # extract experiment name which is top level directory
    exp_name = results_dir.parts[-1]

    # create results dir for experiment
    out_dir = utils.full_path(os.path.join(args['out_dir'], exp_name))
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    
    all_data = []
    for launch_file_dir in results_dir.iterdir():
        # each launch file directory has many individual
        # architecture training logs
        for arch_dir in launch_file_dir.iterdir():
            train_log_path = os.path.join(arch_dir, 'train_log.json')
            if not os.path.exists(train_log_path):
                print(f'{train_log_path} does not exist. Ignoring...')
                continue
            else:
                data = get_params_and_ppl(train_log_path)
                if data:
                    all_data.append(data)

    assert len(all_data) > 0

    all_valid_ppls = [data.valid_ppl for data in all_data]
    # negate params since it is inversely related to perplexity
    all_decoder_params = [-data.decoder_params for data in all_data] 
    all_total_params = [-data.total_params for data in all_data]
    scorr_decoder, pvalue = spearmanr(all_valid_ppls, all_decoder_params)
    scorr_total, pvalue = spearmanr(all_valid_ppls, all_total_params)

    print(f'Total archs used {len(all_total_params)}')
    print(f'Spearman with decoder params proxy {scorr_decoder}')
    print(f'Spearman with total params proxy {scorr_total}')
