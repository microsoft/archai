import os
import yaml 
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from itertools import cycle
from cycler import cycler
from collections import OrderedDict
import math as ma
import yaml
import argparse

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def parse_raw_data(root_exp_folder:str, exp_list:List[str])->Dict:
    data = {}
    for exp in exp_list:
        exp_full_path = os.path.join(root_exp_folder, exp)
        with open(os.path.join(exp_full_path, 'raw_data.yaml')) as f:
            data[exp] = yaml.load(f, Loader=yaml.Loader)

    return data

def main():
    parser = argparse.ArgumentParser(description='Cross Experiment Random Search Plots')
    parser.add_argument('--dataset', type=str, default='natsbench_cifar10',
                        help='dataset on which experiments have been run')
    parser.add_argument('--conf-location', type=str, default='scripts/reports/fastarchrank_plots/cross_random_search.yaml', 
                        help='location of conf file')
    args, extra_args = parser.parse_known_args()

    with open(args.conf_location, 'r') as f:
        conf_data = yaml.load(f, Loader=yaml.Loader)

    exp_folder = conf_data['exp_folder']

    far_exp_list = list(conf_data[args.dataset]['fastarchrank'].keys())
    reg_exp_list = list(conf_data[args.dataset]['regular'].keys())
                            
    # parse raw data from all processed experiments
    far_data = parse_raw_data(exp_folder, far_exp_list)
    reg_data = parse_raw_data(exp_folder, reg_exp_list)

    fig = go.Figure()
    for key in far_data.keys():
        legend_name = conf_data[args.dataset]['fastarchrank'][key]
        marker_color = conf_data[args.dataset]['colors']['fastarchrank']
        fig.add_trace(go.Scatter(x=[far_data[key]['avg_duration']], 
                            y=[far_data[key]['avg_max_acc']], 
                            name=legend_name, mode='markers', 
                            marker_color=marker_color,
                            showlegend=True))
    for key in reg_data.keys():
        legend_name = conf_data[args.dataset]['regular'][key]
        marker_color = conf_data[args.dataset]['colors']['regular']    
        fig.add_trace(go.Scatter(x=[reg_data[key]['avg_duration']], 
                            y=[reg_data[key]['avg_max_acc']], 
                            name=legend_name, mode='markers', 
                            marker_color=marker_color,
                            showlegend=True))

    fig.update_layout(title_text="Duration vs. Max. Accuracy Random Search")
    fig.show()
    

    print('dummy')


if __name__ == '__main__':
    main()


