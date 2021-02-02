import os
import yaml 
import matplotlib.pyplot as plt
import numpy as np
import random


def main():

    exp_folder = 'C:\\Users\\dedey\\archai_experiment_reports'
    
    exp_list = ['ft_fb2048_ftlr1.5_fte5_ct256_ftt0.6', \
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

    # parse raw data from all processed experiments
    data = {}
    for exp in exp_list:
        exp_full_path = os.path.join(exp_folder, exp)
        with open(os.path.join(exp_full_path, 'raw_data.yaml')) as f:
            data[exp] = yaml.load(f, Loader=yaml.Loader)


    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(exp_list)*2)]

    # plot spearman correlation vs. top percent of architectures
    plt.clf()
    legend_labels = []

    for i, key in enumerate(data.keys()):
        plt.plot(data[key]['top_percents'], data[key]['spe_freeze'], marker='*', mfc='red', ms=10, color=colors[random.choice(range(len(exp_list)*2))])
        legend_labels.append(key + '_freezetrain')
        plt.plot(data[key]['top_percents'], data[key]['spe_naswot'], marker='.', mfc='blue', ms=10, linestyle='--', color=colors[i])
        legend_labels.append(key + '_naswot')

        # annotate the freezetrain data points with time information
        for j, tp in enumerate(data[key]['top_percents']):
            duration = data[key]['spe_freeze'][j]
            duration_str = f'{duration:0.1f}'
            plt.annotate(duration_str, (tp, data[key]['spe_freeze'][j]))

    plt.ylim((-1.0, 1.0))
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Spearman Correlation')
    plt.legend(labels=legend_labels)
    plt.grid()
    plt.show()
    savename = os.path.join(exp_folder, f'aggregate_spe.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    # plot timing information vs. top percent of architectures
    plt.clf()
    time_legend_labels = []

    for key in data.keys():
        plt.errorbar(data[key]['top_percents'], data[key]['freeze_times_avg'], yerr=np.array(data[key]['freeze_times_std'])/2, marker='*', mfc='red', ms=5)    
        time_legend_labels.append(key + '_freezetrain')
        
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Avg. Duration (s)')
    plt.legend(labels=time_legend_labels)
    plt.grid()
    plt.show()
    savename = os.path.join(exp_folder, f'aggregate_duration.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')



if __name__ == '__main__':
    main()


