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

ZERO_COST_MEASURES = ['fisher', 'grad_norm', 'grasp', 'jacob_cov', 'plain', 'snip', 'synflow', 'synflow_bn']


def parse_raw_data(root_exp_folder:str, exp_list:List[str])->Dict:
    data = {}
    for exp in exp_list:
        exp_full_path = os.path.join(root_exp_folder, exp)
        with open(os.path.join(exp_full_path, 'raw_data.yaml')) as f:
            data[exp] = yaml.load(f, Loader=yaml.Loader)

    return data

def main():
    parser = argparse.ArgumentParser(description='Cross Experiment Plots')
    parser.add_argument('--dataset', type=str, default='nasbench101',
                        help='dataset on which experiments have been run')
    parser.add_argument('--conf-location', type=str, default='scripts/reports/proxynas_plots/cross_exp_conf.yaml', 
                        help='location of conf file')
    args, extra_args = parser.parse_known_args()

    with open(args.conf_location, 'r') as f:
        conf_data = yaml.load(f, Loader=yaml.Loader)

    exp_folder = conf_data['exp_folder']

    exp_list = list(conf_data[args.dataset]['freezetrain'].keys())
    shortreg_exp_list = list(conf_data[args.dataset]['shortreg'].keys())
                            
    # parse raw data from all processed experiments
    data = parse_raw_data(exp_folder, exp_list)
    shortreg_data = parse_raw_data(exp_folder, shortreg_exp_list)

    # optionally parse zero cost data if it exists
    zero_cost_exp_list = None
    zero_cost_data = None
    try:
        zero_cost_exp_list = list(conf_data[args.dataset]['zero_cost'].keys())
        zero_cost_data = parse_raw_data(exp_folder, zero_cost_exp_list)
    except:
        print(f'zero cost data was not found for {args.dataset}')

    
    # plot spe and common ratio vs. time per top percent of architectures
    # ------------------------------------------------------------

    # assuming that all experiments are reporting on the same 
    # values and length of top_percents
    # get one top_percents list 
    top_percents = data[exp_list[0]]['top_percents']

    tp_info = OrderedDict()
    for i, tp in enumerate(top_percents):
        # go through all methods and get duration, spe for this top percent
        this_tp_info = OrderedDict()
        for key in data.keys():            
            exp_name = key
            assert tp == data[key]['top_percents'][i]
            this_tp_info[exp_name] = (data[key]['freeze_times_avg'][i], data[key]['spe_freeze'][i], data[key]['freeze_ratio_common'][i], data[key]['freeze_times_stderr'][i])

        # get zero cost measures
        if zero_cost_data:
            for j, key in enumerate(zero_cost_data.keys()):
                assert tp == zero_cost_data[key]['top_percents'][i]
                for measure in ZERO_COST_MEASURES:
                    basename = conf_data[args.dataset]['zero_cost'][key]
                    spe_name = measure + '_spe'
                    cr_name = measure + '_ratio_common'
                    this_tp_info[basename + ' ' + measure] = (0.0, zero_cost_data[key][spe_name][i], zero_cost_data[key][cr_name][i])

        # get shortreg
        for key in shortreg_data.keys():
            exp_name = key
            assert tp == shortreg_data[key]['top_percents'][i]
            this_tp_info[exp_name] = (shortreg_data[key]['shortreg_times_avg'][i], shortreg_data[key]['spe_shortreg'][i], shortreg_data[key]['shortreg_ratio_common'][i], shortreg_data[key]['shortreg_times_stderr'][i])

        tp_info[tp] = this_tp_info

    # now plot each top percent time vs. spe and common ratio
    #---------------------------------------------------------
    num_plots = len(tp_info)
    num_plots_per_row = 5
    num_plots_per_col = ma.ceil(num_plots / num_plots_per_row)
    subplot_titles = [f'Top {x} %' for x in tp_info.keys()]
    fig = make_subplots(rows=num_plots_per_row, cols=num_plots_per_col, subplot_titles=subplot_titles, shared_yaxes=True)
    fig_cr = make_subplots(rows=num_plots_per_row, cols=num_plots_per_col, subplot_titles=subplot_titles, shared_yaxes=True)

    for ind, tp_key in enumerate(tp_info.keys()):
        counter = 0
        counter_reg = 0
        counter_zero = 0
        for exp in tp_info[tp_key].keys():
            duration = tp_info[tp_key][exp][0]
            spe = tp_info[tp_key][exp][1]
            cr = tp_info[tp_key][exp][2]

            # stderr info for durations
            error_x = None
            if len(tp_info[tp_key][exp]) > 3:
                duration_stderr = tp_info[tp_key][exp][3]
                error_x = dict(type='data', array=[duration_stderr], visible=True, thickness=1, width=0)

            if exp in exp_list:
                marker = counter
                marker_color = conf_data[args.dataset]['colors']['freezetrain']
                legend_text = conf_data[args.dataset]['freezetrain'][exp]
                counter += 1
            elif exp in shortreg_exp_list:
                marker = counter_reg
                marker_color = conf_data[args.dataset]['colors']['shortreg']
                legend_text = conf_data[args.dataset]['shortreg'][exp]
                counter_reg += 1
            else:
                # we modify the exp name for zero cost as it has many measures in it
                marker = counter_zero
                marker_color = conf_data[args.dataset]['colors']['zero_cost']
                legend_text = exp
                counter_zero += 1
                
            row_num = ma.floor(ind/num_plots_per_col) + 1
            col_num = ind % num_plots_per_col + 1
            showlegend = True if ind == 0 else False
            
            fig.add_trace(go.Scatter(x=[duration], error_x=error_x, y=[spe], mode='markers', name=legend_text, 
                            marker_symbol=marker, marker_color=marker_color, showlegend=showlegend, text=exp),  
                        row=row_num, col=col_num)
            
            fig_cr.add_trace(go.Scatter(x=[duration], error_x=error_x, y=[cr], mode='markers', name=legend_text, 
                            marker_symbol=marker, marker_color=marker_color, showlegend=showlegend, text=exp),  
                        row=row_num, col=col_num)

    fig.update_layout(title_text="Duration vs. Spearman Rank Correlation vs. Top %")
    savename = os.path.join(exp_folder, f'{args.dataset}_duration_vs_spe_vs_top_percent.html')
    fig.write_html(savename)
    fig.show()
    

    fig_cr.update_layout(title_text="Duration vs. Common Ratio vs. Top %")
    savename = os.path.join(exp_folder, f'{args.dataset}_duration_vs_common_ratio_vs_top_percent.html')
    fig_cr.write_html(savename)
    fig_cr.show()

    # now plot each top percent time vs. spe and common ratio (for publication)
    # ----------------------------------------------------------------------------
    num_plots = 6
    num_plots_per_row = 6
    num_plots_per_col = 2
    keys_to_plot = [10, 20, 30, 40, 50, 100]

    subplot_titles = []
    for x in keys_to_plot:
        title = f'Top {x} %'
        for rep in range(2):
            subplot_titles.append(title)

    fig_paper = make_subplots(rows=num_plots_per_row, cols=num_plots_per_col, subplot_titles=subplot_titles, shared_yaxes=False)
    #fig_cr_paper = make_subplots(rows=num_plots_per_row, cols=num_plots_per_col, subplot_titles=subplot_titles, shared_yaxes=True)

    for ind, tp_key in enumerate(keys_to_plot):
        counter = 0
        counter_reg = 0
        counter_zero = 0
        for exp in tp_info[tp_key].keys():
            duration = tp_info[tp_key][exp][0]
            spe = tp_info[tp_key][exp][1]
            cr = tp_info[tp_key][exp][2]

            # stderr info for durations
            error_x = None
            if len(tp_info[tp_key][exp]) > 3:
                duration_stderr = tp_info[tp_key][exp][3]
                error_x = dict(type='data', array=[duration_stderr], visible=True, thickness=1, width=0)

            if exp in exp_list:
                marker = counter
                marker_color = conf_data[args.dataset]['colors']['freezetrain']
                legend_text = conf_data[args.dataset]['freezetrain'][exp]
                counter += 1
            elif exp in shortreg_exp_list:
                marker = counter_reg
                marker_color = conf_data[args.dataset]['colors']['shortreg']
                legend_text = conf_data[args.dataset]['shortreg'][exp]
                counter_reg += 1
            else:
                # we modify the exp name for zero cost as it has many measures in it
                marker = counter_zero
                marker_color = conf_data[args.dataset]['colors']['zero_cost']
                legend_text = exp
                counter_zero += 1

            row_num = ind + 1
            col_num = 1
            showlegend = True if ind == 0 else False
            fig_paper.add_trace(go.Scatter(x=[duration], error_x=error_x, y=[spe], mode='markers', name=legend_text, 
                            marker_symbol=marker, marker_color=marker_color, showlegend=showlegend, text=exp),  
                        row=row_num, col=col_num)
            #fig.update_xaxes(title_text="Duration (s)", row=row_num, col=col_num)
            #fig.update_yaxes(title_text="SPE", row=row_num, col=col_num)

            col_num = 2
            showlegend = False
            fig_paper.add_trace(go.Scatter(x=[duration], error_x=error_x, y=[cr], mode='markers', name=legend_text, 
                            marker_symbol=marker, marker_color=marker_color, showlegend=showlegend, text=exp),  
                        row=row_num, col=col_num)

    # Update each subplot's x and y axes labels
    fig_paper.update_xaxes(title_text="Duration (s)", row=6, col=1)
    fig_paper.update_xaxes(title_text="Duration (s)", row=6, col=2)

    fig_paper.update_yaxes(title_text="Spearman's Corr.", row=1, col=1)
    fig_paper.update_yaxes(title_text="Common Ratio", row=1, col=2)

    fig_paper.update_layout(font=dict(
        size=16,
    ))
    
    savename_html = os.path.join(exp_folder, f'{args.dataset}_duration_vs_spe_vs_cr_vs_top_percent_PAPER.html')
    fig_paper.write_html(savename_html)

    savename_pdf = os.path.join(exp_folder, f'{args.dataset}_duration_vs_spe_vs_cr_vs_top_percent_PAPER.pdf')
    fig_paper.write_image(savename_pdf, engine="kaleido", width=1500, height=1500, scale=1)
    fig_paper.show()
    
    # plot timing information vs. top percent of architectures
    # ------------------------------------------------------------
    fig_time = go.Figure()
    for key in data.keys():
        fig_time.add_trace(go.Scatter(x=data[key]['top_percents'], y=data[key]['freeze_times_avg'], 
                            error_y=dict(type='data', array=np.array(data[key]['freeze_times_std'])/2, 
                            visible=True), name=key)
                            )
    fig_time.update_layout(title="Duration vs. Top Percent of Architectures", 
                            xaxis_title='Top Percent of Architectures', 
                            yaxis_title='Avg. duration (s)')
    savename = os.path.join(exp_folder, f'{args.dataset}_duration_vs_top_percent.html')
    fig_time.write_html(savename)
    fig_time.show()
    



if __name__ == '__main__':
    main()


