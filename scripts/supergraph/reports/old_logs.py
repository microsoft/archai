# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Dict, Type
import glob
import os
import pathlib

from runstats import Statistics

def main():
    parser = argparse.ArgumentParser(description='NAS E2E Runs')
    parser.add_argument('--logdir', type=str, default='D:\\logdir\\azure\\random_cifar_test',
                        help='folder with logs')
    args, extra_args = parser.parse_known_args()

    lines = []
    top1s=[]
    for filepath in pathlib.Path(args.logdir).rglob('logs.log'):
        epoch = 0
        for line in pathlib.Path(filepath).read_text().splitlines():
            if '[eval_test] Epoch: [  1/1] ' in line:
                top1s.append(Statistics())
                top1 = float(line.strip().split('(')[-1].split(',')[0].split('%')[0].strip())/100.0
                lines.append(f'{epoch}\t{top1}\t{str(filepath)}')
                top1s[epoch].push(top1)
                epoch += 1
    pathlib.Path(os.path.join(args.logdir, 'summary.tsv')).write_text('\n'.join(lines))

    stat_lines = ['epoch\tmean\tstddev\tcount']
    for i,top1 in enumerate(top1s):
        stat_lines.append(f'{i}\t{top1.mean()}\t{top1.stddev() if len(top1)>1 else float("NaN")}\t{len(top1)}')
    pathlib.Path(os.path.join(args.logdir, 'summary_stats.tsv')).write_text('\n'.join(stat_lines))

if __name__ == '__main__':
    main()
