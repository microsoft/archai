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
    parser.add_argument('--list-of-npys', '-l', type=str,
                        default=r'~/logdir/proxynas_test_0001',
                        help='txt file containing list of full path to npy files')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    # load list of npys
    with open(args.list_of_npys, 'r') as f:
        list_data = f.readlines()

    # create results dir for experiment
    out_dir = os.path.join(args.out_dir)
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # load all the npy files
    npys = []
    for l in list_data:
        npys.append(np.load(l.rstrip()))

    avg_heatmap = sum(npys)/len(npys)

    # agreed upon order.
    ZEROCOST_MEASURES_PF = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'synflow', 'params', 'flops', 'gt']
    
    fig = px.imshow(avg_heatmap, text_auto="0.1f", x=ZEROCOST_MEASURES_PF, y=ZEROCOST_MEASURES_PF)
    fig.update_layout(font=dict(size=36)) # font size
    savename_html = os.path.join(out_dir, f'avg_all_pairs_zc_spe.html')
    savename_png = os.path.join(out_dir, f'avg_all_pairs_zc_spe.png')
    fig.write_html(savename_html)
    fig.write_image(savename_png, width=1500, height=1500, scale=1)



if __name__ == '__main__':
    main()
