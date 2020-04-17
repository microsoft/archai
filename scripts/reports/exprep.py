import argparse
from typing import Dict, List, Type, Iterator, Tuple
import glob
import os
import pathlib
from collections import OrderedDict
import yaml
from inspect import getsourcefile

from runstats import Statistics

from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger


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
            if not path_key in collated.items():
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
        s += f'±{stat.stddev():.4f}'
    return s

def epoch_nodes_lines(node_path:str, logs_epochs_nodes:List[OrderedDict])->List[str]:
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

    lines = []
    lines.append(f'### Epochs: {node_path}')
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

    return lines


def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--results-dir', '-d', type=str, default=r'D:\GitHubSrc\archaiphilly\phillytools\darts_baseline_20200411',
                        help='folder with experiment results from pt')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    results_dir = pathlib.Path(utils.full_path(args.results_dir))
    print(f'results_dir: {results_dir}')

    exp_name = results_dir.stem

    out_dir = utils.full_path(os.path.join(args.out_dir, exp_name))
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    logs = []

    job_count = 0
    for job_dir in results_dir.iterdir():
        job_count += 1
        for subdir in job_dir.iterdir():
            is_search = subdir.stem.endswith('_search')
            is_eval = subdir.stem.endswith('_eval')
            assert is_search or is_eval

            logs_filepath = os.path.join(str(subdir), 'logs.yaml')
            if os.path.isfile(logs_filepath):
                with open(logs_filepath, 'r') as f:
                    logs.append(yaml.load(f, Loader=yaml.Loader))

    collated = collate_epoch_nodes(logs)
    lines = []
    for node_path, logs_epochs_nodes in collated.items():
        lines += epoch_nodes_lines(node_path, logs_epochs_nodes)
        lines.append('')

    epochs_info = '\n'.join(lines)

    script_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
    template = pathlib.Path(os.path.join(script_dir, 'template.md')).read_text()
    report = template.format(**vars())
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write(report)



if __name__ == '__main__':
    main()