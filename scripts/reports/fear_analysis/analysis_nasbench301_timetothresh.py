import json
import argparse
import os
from typing import List
from tqdm import tqdm

import plotly.graph_objects as go


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

    timetothresh_vs_test_acc = []

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
                time_to_thresh = per_epoch_time * num_epochs_to_thresh 
                timetothresh_vs_test_acc.append((time_to_thresh, test_acc))

    # plot
    fig = go.Figure()
    xs = [timetothresh for timetothresh, testacc in timetothresh_vs_test_acc]
    ys = [testacc for timetothresh, testacc in timetothresh_vs_test_acc]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers'))
    fig.update_layout(xaxis_title='Time to reach threshold train accuracy (s)', 
                      yaxis_title='Final Accuracy')
    savename_html = os.path.join(args.out_dir, 'nasbench301_timetothresh_vs_testacc.html')
    fig.write_html(savename_html)
    fig.show()



if __name__ == '__main__':
    main()