# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path += [os.path.join(SCRIPT_DIR, '..', 'util')]
from shell import Shell


def extract_dlc_metrics(csv):
    TOTAL_PARAMS = "Total parameters:"
    TOTAL_MACS = "Total MACs per inference:"
    for line in csv.split('\n'):
        if TOTAL_MACS in line:
            i = line.index(TOTAL_MACS)
            macs = line[i + len(TOTAL_MACS):].strip().split(' ')[0].strip()
            if macs.endswith('M'):
                macs = macs[:-1] + "000000"
            if macs.endswith('B'):
                macs = macs[:-1] + "000000000"
            macs = int(macs)
            break
        if TOTAL_PARAMS in line:
            i = line.index(TOTAL_PARAMS)
            params = line[i + len(TOTAL_PARAMS):].split('(')[0].strip()
            params = int(params)
    return (macs, params)


def get_dlc_metrics(dlc_file):
    from olive.snpe import (
        SNPEModelMetrics
    )

    helper = SNPEModelMetrics(dlc_file)
    csv_data = helper.execute()
    macs, params = extract_dlc_metrics(csv_data)
    return (csv_data, macs, params)
