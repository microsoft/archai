# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys


def get_dlc_metrics(dlc_file):
    from olive.snpe.tools.dev import get_dlc_info, get_dlc_metrics
    csv_data = get_dlc_info(dlc_file)
    info = get_dlc_metrics(dlc_file)
    return (csv_data, info['macs'], info['parameters'])
