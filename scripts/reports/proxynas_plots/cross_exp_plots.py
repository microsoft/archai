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
                'ft_fb256_ftlr0.1_fte10_ct256_ftt0.6', \
                'ft_fb1024_ftlr0.1_fte5_ct256_ftt0.6_c9', \
                'ft_fb1024_ftlr0.1_fte10_ct256_ftt0.6_c9', \
                'ft_fb512_ftlr0.1_fte5_ct256_ftt0.6_c9', \
                'ft_fb512_ftlr0.1_fte10_ct256_ftt0.6_c9', \
                'ft_fb256_ftlr0.1_fte5_ct256_ftt0.6_c9', \
                'ft_fb256_ftlr0.1_fte10_ct256_ftt0.6_c9']

    exp_list = master_exp_list

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

    # exp_list = ['ft_fb2048_ftlr1.5_fte5_ct256_ftt0.6', \
    #             'ft_fb256_ftlr1.5_fte5_ct256_ftt0.6', \
    #             'ft_fb1024_ftlr0.1_fte5_ct256_ftt0.6', \
    #             'ft_fb512_ftlr0.1_fte5_ct256_ftt0.6']


    shortreg_exp_list = ['nb_reg_b1024_e01', \
                         'nb_reg_b1024_e02', \
                         'nb_reg_b1024_e04', \
                         'nb_reg_b1024_e06', \
                         'nb_reg_b1024_e08', \
                         'nb_reg_b1024_e10', \
                         'nb_reg_b512_e01', \
                         'nb_reg_b512_e02', \
                         'nb_reg_b512_e04', \
                         'nb_reg_b512_e06', \
                         'nb_reg_b512_e08', \
                         'nb_reg_b512_e10' ]
                        
    # parse raw data from all processed experiments
    data = parse_raw_data(exp_folder, exp_list)
    shortreg_data = parse_raw_data(exp_folder, shortreg_exp_list)

    # collect linestyles and colors to create distinguishable plots
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(exp_list)*2)]
    linestyles = ['solid', 'dashdot', 'dotted', 'dashed']
    #markers = ['.', 'v', '^', '<', '>', '1', 's', 'p', '*', '+', 'x', 'X', 'D', 'd']
    markers = ["$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$", "$10$", "$11$", "$12$", "$13$", "$14$", "$15$", "$16$", "$17$", "$18$", "$19$", "$20$", "$21$", "$22$", "$23$", "$24$", "$25$", "$26$", "$27$", "$28$"]
    mathy_markers = ["$a$", "$b$", "$c$", "$d$", "$e$", "$f$", "$g$", "$h$", "$i$", "$j$", "$k$", "$l$", "$m$", "$n$", "$o$", "$p$", "$q$", "$r$", "$s$", "$t$", "$u$", "$v$", "$w$", "$x$", "$y$", "$z$"]
    
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
    counter = 0
    for i, key in enumerate(shortreg_data.keys()):
        plt.plot(shortreg_data[key]['top_percents'], shortreg_data[key]['spe_shortreg'], marker = mathy_markers[counter], mfc='green', ms=10)
        counter += 1
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
            this_tp_info[exp_name] = (data[key]['freeze_times_avg'][i], data[key]['spe_freeze'][i], data[key]['freeze_ratio_common'][i])

        # get only one naswot    
        for j, key in enumerate(data.keys()):
            exp_name = key + '_naswot'
            assert tp == data[key]['top_percents'][i]
            this_tp_info[exp_name] = (0.0, data[key]['spe_naswot'][i], data[key]['naswot_ratio_common'][i])
            if j == 0:
                break

        # get shortreg
        for key in shortreg_data.keys():
            exp_name = key
            assert tp == shortreg_data[key]['top_percents'][i]
            this_tp_info[exp_name] = (shortreg_data[key]['shortreg_times_avg'][i], shortreg_data[key]['spe_shortreg'][i], shortreg_data[key]['shortreg_ratio_common'][i])

        tp_info[tp] = this_tp_info

    # now plot each top percent time vs. spe  
    fig, axs = plt.subplots(5, 10)
    handles = None
    labels = None
    for tp_key, ax in zip(tp_info.keys(), axs.flat):
        counter = 0
        counter_reg = 0
        for exp in tp_info[tp_key].keys():
            duration = tp_info[tp_key][exp][0]
            spe = tp_info[tp_key][exp][1]

            if 'ft_fb' in exp:
                marker = markers[counter]
                counter += 1
            elif 'nb_reg' in exp:
                marker = mathy_markers[counter_reg]
                counter_reg += 1
            else:
                raise NotImplementedError

            ax.scatter(duration, spe, label=exp, marker=marker)
            ax.set_title(str(tp_key))
            #ax.set(xlabel='Duration (s)', ylabel='SPE')
            ax.set_ylim([0, 1])
            
    
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')


    # now plot each top percent time vs. common ratio
    fig, axs = plt.subplots(5, 10)
    handles = None
    labels = None
    for tp_key, ax in zip(tp_info.keys(), axs.flat):
        counter = 0
        counter_reg = 0
        for exp in tp_info[tp_key].keys():
            duration = tp_info[tp_key][exp][0]
            common_ratio = tp_info[tp_key][exp][2]

            if 'ft_fb' in exp:
                marker = markers[counter]
                counter += 1
            elif 'nb_reg' in exp:
                marker = mathy_markers[counter_reg]
                counter_reg += 1
            else:
                raise NotImplementedError

            ax.scatter(duration, common_ratio, label=exp, marker=marker)
            ax.set_title(str(tp_key))
            #ax.set(xlabel='Duration (s)', ylabel='Common Ratio')
            ax.set_ylim([0, 1])
            
    
    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')


    
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


