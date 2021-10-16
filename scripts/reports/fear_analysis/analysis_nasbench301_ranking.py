import json
import argparse
import os
from typing import List
from tqdm import tqdm

from scipy.stats import kendalltau, spearmanr

import plotly.graph_objects as go


def find_train_thresh_epochs(train_acc:List[float], train_thresh:float)->int:
    for i, t in enumerate(train_acc):
        if t >= train_thresh:
            return i 


def main():
    parser = argparse.ArgumentParser(description='Nasbench301 Ranking Experiments')
    parser.add_argument('--nb301-logs-dir', '-d', type=str, help='folder with nasbench301 architecture training logs')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports', help='folder to output reports')
    args, extra_args = parser.parse_known_args()


    all_test_acc = []
    all_train_acc_end = []
    all_train_loss_end = []
    all_val_acc_end = []

    # collect all the json file names in the log dir recursively
    for root, dir, files in os.walk(args.nb301_logs_dir):
        for name in tqdm(files):
            log_name = os.path.join(root, name) 
            with open(log_name, 'r') as f:
                log_data = json.load(f)
                test_acc = log_data['test_accuracy']
                train_acc_end = log_data['learning_curves']['Train/train_accuracy'][-1]
                train_loss_end = log_data['learning_curves']['Train/train_loss'][-1]
                val_acc_end = log_data['learning_curves']['Train/val_accuracy'][-1]

                all_test_acc.append(test_acc)
                all_train_acc_end.append(train_acc_end)
                all_train_loss_end.append(train_loss_end)
                all_val_acc_end.append(val_acc_end)

    # negate the training loss for ranking purposes
    all_train_loss_neg_end = [-x for x in all_train_loss_end]
                
    train_acc_spe, _ = spearmanr(all_test_acc, all_train_acc_end)
    train_loss_spe, _ = spearmanr(all_test_acc, all_train_loss_neg_end)
    val_loss_spe, _ = spearmanr(all_test_acc, all_val_acc_end)

    print(f'Spearman training accuracy end: {train_acc_spe}')
    print(f'Spearman training loss end: {train_loss_spe}')
    print(f'Spearman val loss end: {val_loss_spe}')

if __name__ == '__main__':
    main()