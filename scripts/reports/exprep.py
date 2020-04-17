import argparse
from typing import Dict, Type
import glob
import os
import pathlib

import yaml

from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger
from os import write

def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--results-dir', '-d', type=str, default=r'D:\GitHubSrc\archaiphilly\phillytools\darts_baseline_20200411',
                        help='folder with experiment results from pt')
    parser.add_argument('--out-dir', '-o', type=str, default=r'reports',
                        help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    results_dir = pathlib.Path(utils.full_path(args.results_dir))
    print(f'results_dir: {results_dir}')

    out_dir = utils.full_path(os.path.join(args.out_dir, results_dir.stem))
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    search_logs, eval_logs = [], []

    for job_dir in results_dir.iterdir():
        for subdir in job_dir.iterdir():
            is_search = subdir.stem.endswith('_search')
            is_eval = subdir.stem.endswith('_eval')
            assert is_search or is_eval
            logs = search_logs if is_search else eval_logs

            logs_filepath = os.path.join(str(subdir), 'logs.yaml')
            if os.path.isfile(logs_filepath):
                with open(logs_filepath, 'r') as f:
                    logs.append(yaml.load(f, Loader=yaml.Loader))

    pass
    # for log in search_logs:
    #     write_epochs()

if __name__ == '__main__':
    main()