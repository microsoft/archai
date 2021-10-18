from collections import defaultdict
import json
import argparse
import os
from typing import List
from tqdm import tqdm
import math as ma
import numpy as np

import plotly.graph_objects as go
from scipy.stats import kendalltau, spearmanr, sem
import statistics



def find_train_thresh_epochs(train_acc:List[float], train_thresh:float)->int:
    for i, t in enumerate(train_acc):
        if t >= train_thresh:
            return i + 1


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
    top_percent_range = range(2, 101, 2)
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






def main():
    parser = argparse.ArgumentParser(description='Nasbench301 time to threshold vs. test accuracy')
    parser.add_argument('--nb301-logs-dir', '-d', type=str, help='folder with nasbench301 architecture training logs')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports', help='folder to output reports')
    args, extra_args = parser.parse_known_args()

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
                num_epochs = len(log_data['learning_curves']['Train/train_accuracy'])
                test_acc = log_data['test_accuracy']
                per_epoch_time = log_data['runtime'] / num_epochs
                num_epochs_to_thresh = find_train_thresh_epochs(log_data['learning_curves']['Train/train_accuracy'], 
                                                                train_thresh)
                # many weak architectures will never reach threshold
                if not num_epochs_to_thresh:
                    continue                
                simulated_stage2_epoch = num_epochs_to_thresh + post_thresh_epochs
                fear_time = per_epoch_time * simulated_stage2_epoch
                try:
                    train_acc_stage2 = log_data['learning_curves']['Train/train_accuracy'][simulated_stage2_epoch]
                except:
                    continue

                all_test_acc.append(test_acc)
                all_fear_end_acc.append(train_acc_stage2)
                all_fear_time.append(fear_time)

                # get training acc at all epochs for regular 
                # evaluation baseline
                for epoch_num, train_acc in enumerate(log_data['learning_curves']['Train/train_accuracy']):
                    all_reg_train_acc[epoch_num].append(train_acc)
                    all_reg_train_time_per_epoch[epoch_num].append((epoch_num + 1) * per_epoch_time)                



    fear_train_acc_spe, _ = spearmanr(all_test_acc, all_fear_end_acc)
    print(f'FEAR Spearman training accuracy: {fear_train_acc_spe}')
    print(f'FEAR avg time: {statistics.mean(all_fear_time)}')

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
        # print(f'Epoch {epoch_num}, spearman {spe}, avg. time: {avg_time} seconds')

    # FEAR rank correlations at top n percent of architectures
    # -------------------------------------------------------------
    fear_results = top_buckets_spearmans(all_reg_evals=all_test_acc,
                                        all_proxy_evals=all_fear_end_acc,
                                        all_proxy_times=all_fear_time)


    # Regular evaluation rank correlations at top n percent of architectures
    # -----------------------------------------------------------------------
    
    

    print('dummy') 





    


if __name__ == '__main__':
    main()