from collections import defaultdict
from enum import Enum
import json
import argparse
import os
from typing import Dict, List
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy.stats import kendalltau, spearmanr, sem
import statistics

SCORERS = {'train_accuracy', 'train_loss', 'train_cross_entropy', 'val_accuracy'}


def plot_spearman_top_percents(results:Dict[str, list], 
                                plotly_fig_handle,
                                legend_text:str,
                                marker_color:str):
    
    for idx, tp in enumerate(results['top_percents']):
        avg_time = results['avg_times'][idx]
        stderr = results['stderr_times'][idx]
        error_x = dict(type='data', array=[stderr], visible=True, thickness=1, width=0)
        spe = results['spes'][idx]
        show_legend = False if idx > 0 else True
        plotly_fig_handle.add_trace(go.Scatter(x=[avg_time],
                                 error_x=error_x, 
                                 y=[spe], 
                                 mode='markers',
                                 name=legend_text,
                                 showlegend=show_legend,
                                 marker_color=marker_color),
                                 row=idx+1, col=1)    
    

def find_train_thresh_epochs(train_acc:List[float], train_thresh:float)->int:
    for i, t in enumerate(train_acc):
        if t >= train_thresh:
            return i + 1


<<<<<<< HEAD
=======
def top_buckets_spearmans(all_reg_evals:List[float],
                        all_proxy_evals:List[float],
                        all_proxy_times:List[float]):

    assert len(all_reg_evals) == len(all_proxy_evals)
    assert len(all_reg_evals) == len(all_proxy_times)
    reg_proxy = [(x, y, z) for x, y, z in zip(all_reg_evals, all_proxy_evals, all_proxy_times)]
    
    # sort in descending order of accuracy of regular evaluation
    reg_proxy.sort(key= lambda x: x[0], reverse=True)

    top_percent_times_avg = []
    top_percent_times_std = []
    top_percent_times_stderr = []

    spe_top_percents = []

    top_percents = []
    top_percent_range = range(10, 101, 10)
    for top_percent in top_percent_range:
        top_percents.append(top_percent)
        num_to_keep = int(ma.floor(len(reg_proxy) * top_percent * 0.01))
        top_percent_reg_proxy_times = reg_proxy[:num_to_keep]
        top_percent_reg = [x[0] for x in top_percent_reg_proxy_times]
        top_percent_proxy = [x[1] for x in top_percent_reg_proxy_times]
        top_percent_proxy_times = [x[2] for x in top_percent_reg_proxy_times]

        top_percent_times_avg.append(np.mean(np.array(top_percent_proxy_times)))
        top_percent_times_std.append(np.std(np.array(top_percent_proxy_times)))
        top_percent_times_stderr.append(sem(np.array(top_percent_proxy_times)))

        spe_proxy, _ = spearmanr(top_percent_reg, top_percent_proxy)
        spe_top_percents.append(spe_proxy)

    results = {
        'top_percents': top_percents,
        'spes': spe_top_percents,
        'avg_times': top_percent_times_avg,
        'std_times': top_percent_times_std,
        'stderr_times': top_percent_times_stderr
    }

    return results



>>>>>>> 17e92924 (Simulation code on DARTS logs nominally working.)
def main():
    parser = argparse.ArgumentParser(description='Nasbench301 time to threshold vs. test accuracy')
    parser.add_argument('--nb301-logs-dir', '-d', type=str, help='folder with nasbench301 architecture training logs')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports', help='folder to output reports')
    parser.add_argument('--scorer', '-s', type=str, default='train_accuracy', 
                help='one of train_accuracy, train_loss, train_cross_entropy, val_accuracy')
    args, extra_args = parser.parse_known_args()

    if args.scorer not in SCORERS:
        raise argparse.ArgumentError
    scorer_key = "Train/" + args.scorer

    # TODO: make these into cmd line arguments
    train_thresh = 60.0
    post_thresh_epochs = 10

    all_test_acc = []
    all_fear_end_acc = []
    all_fear_time = []

    all_reg_train_acc = defaultdict(list)
    all_reg_train_time_per_epoch = defaultdict(list)
    
    # collect all the json file names in the log dir recursively
    for root, dir, files in os.walk(args.nb301_logs_dir):
        for name in tqdm(files):
            log_name = os.path.join(root, name) 
            with open(log_name, 'r') as f:
                log_data = json.load(f)
                num_epochs = len(log_data['learning_curves'][scorer_key])
                test_acc = log_data['test_accuracy']
                per_epoch_time = log_data['runtime'] / num_epochs
                num_epochs_to_thresh = find_train_thresh_epochs(log_data['learning_curves'][scorer_key], 
                                                                train_thresh)
                # many weak architectures will never reach threshold
                if not num_epochs_to_thresh:
                    continue                
                simulated_stage2_epoch = num_epochs_to_thresh + post_thresh_epochs
                fear_time = per_epoch_time * simulated_stage2_epoch
                try:
                    train_acc_stage2 = log_data['learning_curves'][scorer_key][simulated_stage2_epoch]
                except:
                    continue

                all_test_acc.append(test_acc)
                all_fear_end_acc.append(train_acc_stage2)
                all_fear_time.append(fear_time)

                # get training acc at all epochs for regular 
                # evaluation baseline
                for epoch_num, train_acc in enumerate(log_data['learning_curves'][scorer_key]):
                    all_reg_train_acc[epoch_num].append(train_acc)
                    all_reg_train_time_per_epoch[epoch_num].append((epoch_num + 1) * per_epoch_time)                


    spes_train_acc_vs_epoch = {}
    avg_time_train_acc_vs_epoch = {}
    for epoch_num, train_accs_epoch in all_reg_train_acc.items():
        if len(train_accs_epoch) != len(all_test_acc):
            continue
        this_spe, _ = spearmanr(all_test_acc, train_accs_epoch)
        spes_train_acc_vs_epoch[epoch_num] = this_spe
        avg_time_train_acc_vs_epoch[epoch_num] = statistics.mean(all_reg_train_time_per_epoch[epoch_num])

    for epoch_num, spe in spes_train_acc_vs_epoch.items():
        avg_time = avg_time_train_acc_vs_epoch[epoch_num]
<<<<<<< HEAD
        print(f'Epoch {epoch_num}, spearman {spe}, avg. time: {avg_time} seconds')                
=======
        # print(f'Epoch {epoch_num}, spearman {spe}, avg. time: {avg_time} seconds')

    # FEAR rank correlations at top n percent of architectures
    # -------------------------------------------------------------
    fear_results = top_buckets_spearmans(all_reg_evals=all_test_acc,
                                        all_proxy_evals=all_fear_end_acc,
                                        all_proxy_times=all_fear_time)

    # picking epoch 10 to plot for regular evaluation
    reg_results = {}
    for epoch_num in all_reg_train_acc.keys():
        all_reg = all_reg_train_acc[epoch_num]
        all_reg_times = all_reg_train_time_per_epoch[epoch_num]
        if len(all_test_acc) != len(all_reg):
            continue
        reg_results[epoch_num] = top_buckets_spearmans(all_reg_evals=all_test_acc,
                                                    all_proxy_evals=all_reg,    
                                                    all_proxy_times=all_reg_times)

    # plot
    num_plots = len(fear_results['top_percents'])
    num_plots_per_row = num_plots
    num_plots_per_col = 1

    subplot_titles = [f'Top {x} %' for x in fear_results['top_percents']]

    fig = make_subplots(rows=num_plots_per_row, 
                            cols=num_plots_per_col, 
                            subplot_titles=subplot_titles, 
                            shared_yaxes=False)
   
    plot_spearman_top_percents(fear_results, fig, 'FEAR', 'red')

    for epoch_num, epoch_num_results in reg_results.items():
        plot_spearman_top_percents(epoch_num_results, fig, f'Regular epochs {epoch_num}', 'blue')

    fig.update_layout(title_text="Duration vs. Spearman Rank Correlation vs. Top %")
    fig.show()
    
    
    # Regular evaluation rank correlations at top n percent of architectures
    # -----------------------------------------------------------------------
    
    

    print('dummy') 
>>>>>>> 17e92924 (Simulation code on DARTS logs nominally working.)





    


if __name__ == '__main__':
    main()