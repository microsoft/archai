import os
import argparse
from typing import Dict, List, Type, Iterator, Tuple
from collections import OrderedDict, defaultdict
from scipy.stats.stats import _two_sample_transform
import yaml
import json
from inspect import getsourcefile
import seaborn as sns
import math as ma
import numpy as np

from scipy.stats import kendalltau, spearmanr, pearsonr

import plotly.express as px
import plotly.graph_objects as go



def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--transnasbench-results-json', '-t', type=str,
                        default=r'~/logdir/proxynas_test_0001',
                        help='full path to json file of transnasbench results')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    # load json file
    with open(args.transnasbench_results_json, 'r') as f:
        json_data = json.load(f)

    # create results dir for experiment
    out_dir = os.path.join(args.out_dir)
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    ZEROCOST_MEASURES_PF = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'synflow', 'params', 'flops', 'gt']
    
    for task in json_data.keys():
        task_data = json_data[task]
        hm = np.zeros((len(ZEROCOST_MEASURES_PF), len(ZEROCOST_MEASURES_PF))) 
        for i, m1 in enumerate(ZEROCOST_MEASURES_PF):
            for j, m2 in enumerate(ZEROCOST_MEASURES_PF):
                # sometimes jacob_cov has a nan here and there. ignore those.
                m1_scores = task_data[m1]
                m2_scores = task_data[m2]
                valid_scores = [x for x in zip(m1_scores, m2_scores) if not ma.isnan(x[0]) and not ma.isnan(x[1])]
                m1_valid = [x[0] for x in valid_scores]
                m2_valid = [x[1] for x in valid_scores]
                spe, _ = spearmanr(m1_valid, m2_valid)
                hm[i][j] = spe
        
        fig = px.imshow(hm, text_auto="0.1f", x=ZEROCOST_MEASURES_PF, y=ZEROCOST_MEASURES_PF)
        fig.update_layout(font=dict(size=36)) # font size
        savename_html = os.path.join(out_dir, f'{task}_all_pairs_zc_spe.html')
        savename_png = os.path.join(out_dir, f'{task}_all_pairs_zc_spe.png')
        fig.write_html(savename_html)
        fig.write_image(savename_png, width=1500, height=1500, scale=1)



if __name__ == '__main__':
    main()
