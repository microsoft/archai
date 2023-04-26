# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import json
import statistics
from status import get_all_status_entities, update_status_entity

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'

STDEV_THRESHOLD = 10     # redo any runs that have a stdev > 10% of the mean.
MAX_COUNT = 100


def find_unsteady_runs(threshold, reset, limit=None):
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    wobbly = []
    # Check standard deviation and if it is more than %5 of the mean then
    # reset the total_inference_avg so it re-runs.
    for e in get_all_status_entities():
        name = e['name']
        if 'total_inference_avg' in e and 'model_date' in e:
            total_inference_avg = json.loads(e['total_inference_avg'])
            if len(total_inference_avg) < 2:
                continue
            stdev = int(statistics.stdev(total_inference_avg))
            mean = int(statistics.mean(total_inference_avg))
            changed = False
            if 'stdev' not in e:
                e['stdev'] = int((stdev * 100) / mean)
                changed = True
            r = int(stdev * 100 / mean)
            if r >= threshold:
                print(f"Found {name}, with mean {mean}, stdev {stdev} which is {r}% of the mean")
                wobbly += [e]

            if changed:
                update_status_entity(e)

    if reset:
        s = sorted(wobbly, key=lambda e: e['model_date'])
        s.reverse()
        if limit:
            print(f"Found {len(s)} wobbly jobs, but limiting reset to the newest {limit} jobs")
            s = s[0:limit]

        for e in s:
            name = e['name']
            print(f"Resetting {name} total_inference_avg={e['total_inference_avg']}...")
            del e['total_inference_avg']
            e['status'] = 'reset'
            update_status_entity(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reset jobs that have a stdev above a given percentage level ' +
        'and optionally reset them so they they run again.')
    parser.add_argument(
        '--threshold', type=int,
        help=f'What percentage stddev to use as threshold (default {STDEV_THRESHOLD}).',
        default=STDEV_THRESHOLD)
    parser.add_argument(
        '--limit', type=int,
        help=f'Maximum number of jobs to reset (default {MAX_COUNT}).',
        default=MAX_COUNT)
    parser.add_argument(
        '--reset',
        help='Reset the runs found to be unsteady so they run again.',
        action="store_true")
    args = parser.parse_args()
    if args.threshold < 1:
        print("### threshold must be greater than 1")
    else:
        find_unsteady_runs(args.threshold, args.reset, args.limit)
