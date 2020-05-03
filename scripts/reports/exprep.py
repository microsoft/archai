import argparse
from typing import Dict, List, Type, Iterator, Tuple
import glob
import os
import pathlib
from collections import OrderedDict
import yaml
from inspect import getsourcefile

from runstats import Statistics

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger
import re


def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--results-dir', '-d', type=str, default=r'D:\GitHubSrc\archaiphilly\phillytools\darts_baseline_20200411',
                        help='folder with experiment results from pt')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    # root dir where all results are stored
    results_dir = pathlib.Path(utils.full_path(args.results_dir))
    print(f'results_dir: {results_dir}')

    # extract experiment name which is top level directory
    exp_name = results_dir.stem

    # create results dir for experiment
    out_dir = utils.full_path(os.path.join(args.out_dir, exp_name))
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # get list of all structured logs for each job
    logs = []
    job_count = 0
    for job_dir in results_dir.iterdir():
        job_count += 1
        for subdir in job_dir.iterdir():
            # currently we expect that each job was ExperimentRunner job which should have
            # _search or _eval folders
            is_search = subdir.stem.endswith('_search')
            is_eval = subdir.stem.endswith('_eval')
            assert is_search or is_eval

            logs_filepath = os.path.join(str(subdir), 'logs.yaml')
            if os.path.isfile(logs_filepath):
                with open(logs_filepath, 'r') as f:
                    logs.append(yaml.load(f, Loader=yaml.Loader))

    collated_logs = collate_epoch_nodes(logs)
    summary_text, details_text = '', ''
    for node_path, logs_epochs_nodes in collated_logs.items():
        collated_epoch_stats = get_epoch_stats(node_path, logs_epochs_nodes)
        summary_text += get_summary_text(out_dir, node_path, collated_epoch_stats)
        details_text += get_details_text(out_dir, node_path, collated_epoch_stats)

    write_report('summary.md', **vars())
    write_report('details.md', **vars())

def epoch_nodes(node:OrderedDict, path=[])->Iterator[Tuple[List[str], OrderedDictLogger]]:
    for k,v in node.items():
        if k == 'epochs' and isinstance(v, OrderedDict) and len(v) and '0' in v:
            yield path, v
        elif isinstance(v, OrderedDict):
            for p, en in epoch_nodes(v, path=path+[k]):
                yield p, en

def collate_epoch_nodes(logs:List[OrderedDict])->Dict[str, List[OrderedDict]]:
    collated = OrderedDict()
    for log in logs:
        for path, epoch_node in epoch_nodes(log):
            path_key = '/'.join(path)
            if not path_key in collated:
                collated[path_key] = []
            v = collated[path_key]
            v.append(epoch_node)
    return collated


class EpochStats:
    def __init__(self) -> None:
        self.start_lr = Statistics()
        self.end_lr = Statistics()
        self.train_fold = FoldStats()
        self.val_fold = FoldStats()

    def update(self, epoch_node:OrderedDict)->None:
        self.start_lr.push(epoch_node['start_lr'])
        self.end_lr.push(epoch_node['train']['end_lr'])
        self.train_fold.update(epoch_node['train'])
        self.val_fold.update(epoch_node['val'])

class FoldStats:
    def __init__(self) -> None:
        self.top1 = Statistics()
        self.top5 = Statistics()
        self.duration = Statistics()
        self.step_time = Statistics()

    def update(self, fold_node:OrderedDict)->None:
        self.top1.push(fold_node['top1'])
        self.top5.push(fold_node['top5'])
        if 'duration' in fold_node:
            self.duration.push(fold_node['duration'])
        if 'step_time' in fold_node:
            self.step_time.push(fold_node['step_time'])

def stat2str(stat:Statistics)->str:
    if len(stat) == 0:
        return '-'
    s = f'{stat.mean():.4f}'
    if len(stat)>1:
        s += f'<sup> ± {stat.stddev():.4f}</sup>'
    return s

def get_epoch_stats(node_path:str, logs_epochs_nodes:List[OrderedDict])->List[EpochStats]:
    epoch_stats = []

    for epochs_node in logs_epochs_nodes:
        for epoch_num, epoch_node in epochs_node.items():
            if not str.isnumeric(epoch_num):
                continue
            epoch_num = int(epoch_num)
            if epoch_num >= len(epoch_stats):
                epoch_stats.append(EpochStats())
            epoch_stat = epoch_stats[epoch_num]
            epoch_stat.update(epoch_node)

    return epoch_stats

def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def get_summary_text(out_dir:str, node_path:str, epoch_stats:List[EpochStats])->str:
    lines = ['','']

    lines.append(f'### Summary: {node_path}')

    plot_filename = get_valid_filename(node_path)+'.png'
    plot_filepath = os.path.join(out_dir, plot_filename)
    plot_epochs(epoch_stats, plot_filepath)

    train_duration = Statistics()
    for epoch_stat in epoch_stats:
        train_duration += epoch_stat.train_fold.duration

    lines.append(f'![]({plot_filename})')

    lines.append(f'Train epoch time: {stat2str(train_duration)}')
    lines.append('')
    for milestone in [35-1, 200-1, 600-1, 1500-1]:
        if len(epoch_stats) >= milestone:
            lines.append(f'{stat2str(epoch_stats[milestone].val_fold.top1)} val top1 @ {milestone} epochs')

    return '\n'.join(lines)

def get_details_text(out_dir:str, node_path:str, epoch_stats:List[EpochStats])->str:
    lines = ['','']
    lines.append(f'### Data: {node_path}')

    lines.append('|Epoch   |Val Top1   |Val Top5   |Train  Top1 |Train Top5   |Train Duration   |Val Duration   |Train Step Time     |Val Step Time   |StartLR   |EndLR   |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')

    for i, epoch_stat in enumerate(epoch_stats):
        line = '|'
        line += str(i) + '|'
        line += stat2str(epoch_stat.val_fold.top1) + '|'
        line += stat2str(epoch_stat.val_fold.top5) + '|'
        line += stat2str(epoch_stat.train_fold.top1) + '|'
        line += stat2str(epoch_stat.train_fold.top5) + '|'
        line += stat2str(epoch_stat.train_fold.duration) + '|'
        line += stat2str(epoch_stat.val_fold.duration) + '|'
        line += stat2str(epoch_stat.train_fold.step_time) + '|'
        line += stat2str(epoch_stat.val_fold.step_time) + '|'
        line += stat2str(epoch_stat.start_lr) + '|'
        line += stat2str(epoch_stat.end_lr) + '|'

        lines.append(line)

    return '\n'.join(lines)

def plot_epochs(epoch_stats:List[EpochStats], filepath:str):
    plt.ioff()
    plt.clf()
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 5)
    with sns.axes_style("darkgrid"):
        metrics = []
        val_top1_means = [es.val_fold.top1.mean() if len(es.val_fold.top1)>0 else 0 for es in epoch_stats]
        val_top1_std = [es.val_fold.top1.stddev() if len(es.val_fold.top1)>1 else 0 for es in epoch_stats]
        val_top1_min = [es.val_fold.top1.minimum() if len(es.val_fold.top1)>0 else 0 for es in epoch_stats]
        val_top1_max = [es.val_fold.top1.maximum() if len(es.val_fold.top1)>0 else 0 for es in epoch_stats]
        metrics.append((val_top1_means, val_top1_std, 'val_top1', val_top1_min, val_top1_max))

        val_top5_means = [es.val_fold.top5.mean() for es in epoch_stats]
        val_top5_std = [es.val_fold.top5.stddev() if len(es.val_fold.top5)>1 else 0 for es in epoch_stats]
        val_top5_min = [es.val_fold.top5.minimum() if len(es.val_fold.top5)>0 else 0 for es in epoch_stats]
        val_top5_max = [es.val_fold.top5.maximum() if len(es.val_fold.top5)>0 else 0 for es in epoch_stats]
        metrics.append((val_top5_means, val_top5_std, 'val_top5', val_top5_min, val_top5_max))

        train_top1_means = [es.train_fold.top1.mean() for es in epoch_stats]
        train_top1_std = [es.train_fold.top1.stddev() if len(es.train_fold.top1)>1 else 0 for es in epoch_stats]
        train_top1_min = [es.train_fold.top1.minimum() if len(es.train_fold.top1)>0 else 0 for es in epoch_stats]
        train_top1_max = [es.train_fold.top1.maximum() if len(es.train_fold.top1)>0 else 0 for es in epoch_stats]
        metrics.append((train_top1_means, train_top1_std, 'train_top1', train_top1_min, train_top1_max))

        train_top5_means = [es.train_fold.top5.mean() for es in epoch_stats]
        train_top5_std = [es.train_fold.top5.stddev() if len(es.train_fold.top5)>1 else 0 for es in epoch_stats]
        train_top5_min = [es.train_fold.top1.minimum() if len(es.train_fold.top5)>0 else 0 for es in epoch_stats]
        train_top5_max = [es.train_fold.top1.maximum() if len(es.train_fold.top5)>0 else 0 for es in epoch_stats]
        metrics.append((train_top5_means, train_top5_std, 'train_top5', train_top5_min, train_top5_max))

        for i, metric in enumerate(metrics):
            ax.plot(range(len(metric[0])), metric[0], label=metric[2], c=clrs[i])
            ax.fill_between(range(len(metric[0])), np.subtract(metric[0], metric[1]),
                            np.add(metric[0], metric[1]),
                            alpha=0.5, facecolor=clrs[i])
            ax.fill_between(range(len(metric[0])), metric[3],
                            metric[4],
                            alpha=0.1, facecolor=clrs[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Metrics')

        ax.legend()
        ax.grid('on')

        # add more ticks
        ax.set_xticks(np.arange(max([len(m) for m in metrics])))
        # remove tick marks
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)

        # change the color of the top and right spines to opaque gray
        # ax.spines['right'].set_color((.8,.8,.8))
        # ax.spines['top'].set_color((.8,.8,.8))

        # tweak the axis labels
        xlab = ax.xaxis.get_label()
        ylab = ax.yaxis.get_label()

        xlab.set_style('italic')
        xlab.set_size(10)
        ylab.set_style('italic')
        ylab.set_size(10)

        # tweak the title
        ttl = ax.title
        ttl.set_weight('bold')
    plt.savefig(filepath)
    plt.close()


def write_report(template_filename:str, **kwargs)->None:
    script_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
    template = pathlib.Path(os.path.join(script_dir, template_filename)).read_text()
    report = template.format(**kwargs)
    with open(os.path.join(kwargs['out_dir'], template_filename), 'w') as f:
        f.write(report)

if __name__ == '__main__':
    main()