import os
import yaml 
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List, Dict
from itertools import cycle
from cycler import cycler
from collections import OrderedDict


def parse_raw_data(root_exp_folder:str, exp_list:List[str])->Dict:
    data = {}
    for exp in exp_list:
        exp_full_path = os.path.join(root_exp_folder, exp)
        with open(os.path.join(exp_full_path, 'raw_data.yaml')) as f:
            data[exp] = yaml.load(f, Loader=yaml.Loader)

    return data

def main():

    
    exp_folder = 'C:\\Users\\dedey\\archai_experiment_reports'
    
    master_exp_list = ['ft_fb2048_ftlr1.5_fte5_ct256_ftt0.6', \
                'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.6', \
                'ft_fb2048_ftlr1.5_fte5_ct256_ftt0.5', \
                'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.5', \
                'ft_fb2048_ftlr1.5_fte5_ct256_ftt0.4', \
                'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.4', \
                'ft_fb2048_ftlr1.5_fte5_ct256_ftt0.3', \
                'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.3', \
                'ft_fb1024_ftlr1.5_fte5_ct256_ftt0.6', \
                'ft_fb1024_ftlr1.5_fte10_ct256_ftt0.6', \
                'ft_fb512_ftlr1.5_fte5_ct256_ftt0.6', \
                'ft_fb512_ftlr1.5_fte10_ct256_ftt0.6', \
                'ft_fb256_ftlr1.5_fte5_ct256_ftt0.6', \
                'ft_fb256_ftlr1.5_fte10_ct256_ftt0.6', \
                'ft_fb1024_ftlr0.1_fte5_ct256_ftt0.6', \
                'ft_fb1024_ftlr0.1_fte10_ct256_ftt0.6', \
                'ft_fb512_ftlr0.1_fte5_ct256_ftt0.6', \
                'ft_fb512_ftlr0.1_fte10_ct256_ftt0.6', \
                'ft_fb256_ftlr0.1_fte5_ct256_ftt0.6', \
                'ft_fb256_ftlr0.1_fte10_ct256_ftt0.6']

    # exp_list = ['ft_fb2048_ftlr1.5_fte5_ct256_ftt0.6', \
    #             'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.6', \
    #             'ft_fb2048_ftlr1.5_fte5_ct256_ftt0.5', \
    #             'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.5', \
    #             'ft_fb2048_ftlr1.5_fte5_ct256_ftt0.4', \
    #             'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.4', \
    #             'ft_fb2048_ftlr1.5_fte5_ct256_ftt0.3', \
    #             'ft_fb2048_ftlr1.5_fte10_ct256_ftt0.3']

    # exp_list = ['ft_fb1024_ftlr1.5_fte5_ct256_ftt0.6', \
    #             'ft_fb1024_ftlr1.5_fte10_ct256_ftt0.6', \
    #             'ft_fb512_ftlr1.5_fte5_ct256_ftt0.6', \
    #             'ft_fb512_ftlr1.5_fte10_ct256_ftt0.6', \
    #             'ft_fb256_ftlr1.5_fte5_ct256_ftt0.6', \
    #             'ft_fb256_ftlr1.5_fte10_ct256_ftt0.6']

    # exp_list = ['ft_fb1024_ftlr0.1_fte5_ct256_ftt0.6', \
    #             'ft_fb1024_ftlr0.1_fte10_ct256_ftt0.6', \
    #             'ft_fb512_ftlr0.1_fte5_ct256_ftt0.6', \
    #             'ft_fb512_ftlr0.1_fte10_ct256_ftt0.6', \
    #             'ft_fb256_ftlr0.1_fte5_ct256_ftt0.6', \
    #             'ft_fb256_ftlr0.1_fte10_ct256_ftt0.6']

    exp_list = ['ft_fb2048_ftlr1.5_fte5_ct256_ftt0.6', \
                'ft_fb256_ftlr1.5_fte5_ct256_ftt0.6', \
                'ft_fb1024_ftlr0.1_fte5_ct256_ftt0.6', \
                'ft_fb512_ftlr0.1_fte5_ct256_ftt0.6']


    shortreg_exp_list = ['nb_reg_b1024_e10', 'nb_reg_b1024_e20', 'nb_reg_b1024_e30']

    # parse raw data from all processed experiments
    data = parse_raw_data(exp_folder, exp_list)
    shortreg_data = parse_raw_data(exp_folder, shortreg_exp_list)

    # collect linestyles and colors to create distinguishable plots
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(exp_list)*2)]
    linestyles = ['solid', 'dashdot', 'dotted', 'dashed']
    markers = ['.', 'v', '^', '<', '>', '1', 's', 'p', '*', '+', 'x', 'X', 'D', 'd']
    
    cc = cycler(color=colors) * cycler(linestyle=linestyles) * cycler(marker=markers)

    # plot spearman correlation vs. top percent of architectures
    # -----------------------------------------------------------
    fig, ax = plt.subplots()
    ax.set_prop_cycle(cc)
    legend_labels = []

    # plot freezetrain and naswot data
    for i, key in enumerate(data.keys()):
        plt.plot(data[key]['top_percents'], data[key]['spe_freeze'], ms=10)
        legend_labels.append(key + '_freezetrain')

        # annotate the freezetrain data points with time information
        for j, tp in enumerate(data[key]['top_percents']):
            duration = data[key]['freeze_times_avg'][j]
            duration_str = f'{duration:0.1f}'
            plt.annotate(duration_str, (tp, data[key]['spe_freeze'][j]))

    # just plot one naswot data to not clutter up the system
    for i, key in enumerate(data.keys()):
        plt.plot(data[key]['top_percents'], data[key]['spe_naswot'], marker='8', mfc='blue', ms=10)
        legend_labels.append(key + '_naswot')
        if i == 0:
            break

    # plot shortreg data
    for i, key in enumerate(shortreg_data.keys()):
        plt.plot(shortreg_data[key]['top_percents'], shortreg_data[key]['spe_shortreg'], marker = '8', mfc='green', ms=10)
        legend_labels.append(key)
    
        # annotate the shortreg data points with time information
        for j, tp in enumerate(shortreg_data[key]['top_percents']):
            duration = shortreg_data[key]['shortreg_times_avg'][j]
            duration_str = f'{duration:0.1f}'
            plt.annotate(duration_str, (tp, shortreg_data[key]['spe_shortreg'][j]))

    
    plt.ylim((-1.0, 1.0))
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Spearman Correlation')
    plt.legend(labels=legend_labels)
    plt.grid()
    #plt.show()
    savename = os.path.join(exp_folder, f'aggregate_spe.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    # plot spe vs. time per top percent of architectures
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
            exp_name = key + '_freezetrain'
            assert tp == data[key]['top_percents'][i]
            this_tp_info[exp_name] = (data[key]['freeze_times_avg'][i], data[key]['spe_freeze'][i])

        # get only one naswot    
        for j, key in enumerate(data.keys()):
            exp_name = key + '_naswot'
            assert tp == data[key]['top_percents'][i]
            this_tp_info[exp_name] = (0.0, data[key]['spe_naswot'][i])
            if j == 0:
                break

        # get shortreg
        for key in shortreg_data.keys():
            exp_name = key
            assert tp == shortreg_data[key]['top_percents'][i]
            this_tp_info[exp_name] = (shortreg_data[key]['shortreg_times_avg'][i], shortreg_data[key]['spe_shortreg'][i])

        tp_info[tp] = this_tp_info

    # now plot each top percent
    cc = cycler(marker=markers)
    fig, axs = plt.subplots(5, 10)
    handles = None
    labels = None
    for tp_key, ax in zip(tp_info.keys(), axs.flat):
        for exp in tp_info[tp_key].keys():
            duration = tp_info[tp_key][exp][0]
            spe = tp_info[tp_key][exp][1]
            #ax.set_prop_cycle(cc)
            ax.scatter(duration, spe, label=exp)
            ax.set_title(str(tp_key))
            #ax.set(xlabel='Duration (s)', ylabel='SPE')
            ax.set_ylim([0, 1])
            
    
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    
    plt.show()


    # plot timing information vs. top percent of architectures
    # ------------------------------------------------------------
    plt.clf()
    time_legend_labels = []

    for key in data.keys():
        plt.errorbar(data[key]['top_percents'], data[key]['freeze_times_avg'], yerr=np.array(data[key]['freeze_times_std'])/2, marker='*', mfc='red', ms=5)    
        time_legend_labels.append(key + '_freezetrain')
        
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Avg. Duration (s)')
    plt.legend(labels=time_legend_labels)
    plt.grid()
    #plt.show()
    savename = os.path.join(exp_folder, f'aggregate_duration.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')



if __name__ == '__main__':
    main()


