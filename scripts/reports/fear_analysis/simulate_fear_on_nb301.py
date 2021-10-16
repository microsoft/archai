from collections import defaultdict
import json
import argparse
import os
from typing import List
from tqdm import tqdm

import plotly.graph_objects as go
from scipy.stats import kendalltau, spearmanr
import statistics



def find_train_thresh_epochs(train_acc:List[float], train_thresh:float)->int:
    for i, t in enumerate(train_acc):
        if t >= train_thresh:
            return i 


def main():
    parser = argparse.ArgumentParser(description='Nasbench301 time to threshold vs. test accuracy')
    parser.add_argument('--nb301-logs-dir', '-d', type=str, help='folder with nasbench301 architecture training logs')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports', help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    train_thresh = 60.0
    post_thresh_epochs = 5

    all_test_acc = []
    all_fear_end_acc = []
    all_fear_time = []

    all_reg_train_acc = defaultdict(list)
    
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



    fear_train_acc_spe, _ = spearmanr(all_test_acc, all_fear_end_acc)
    print(f'FEAR Spearman training accuracy: {fear_train_acc_spe}')
    print(f'FEAR avg time: {statistics.mean(all_fear_time)}')

    spes_train_acc_vs_epoch = {}
    for epoch_num, train_accs_epoch in all_reg_train_acc.items():
        if len(train_accs_epoch) != len(all_test_acc):
            continue
        this_spe, _ = spearmanr(all_test_acc, train_accs_epoch)
        spes_train_acc_vs_epoch[epoch_num] = this_spe

    for epoch_num, spe in spes_train_acc_vs_epoch.items():
        print(f'Epoch {epoch_num}, spearman {spe}')                





    


if __name__ == '__main__':
    main()