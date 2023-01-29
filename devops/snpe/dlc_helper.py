# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path += [os.path.join(SCRIPT_DIR, '..', 'util')]
from shell import Shell
from  test_snpe import add_snpe_env

def view_model(dlc_file):
    """ Run the snpe-dlc-viewer and return the filename containing the html output """
    dir = os.path.dirname(dlc_file)
    filename = os.path.basename(dlc_file)
    basename = os.path.splitext(filename)[0]
    html_file = os.path.join(dir, basename + ".html")
    shell = Shell()
    command = f"snpe-dlc-viewer -i {dlc_file} -s {html_file}"
    shell.run(os.getcwd(), add_snpe_env(command), True)
    return html_file


def get_dlc_metrics(html):
    """ Read the HTML output from snpe-dlc-viewer and return the macs and total params """
    TOTAL_MACS = "Total Macs</td><td>"
    TOTAL_PARAMS = "Total Params</td><td>"
    macs = None
    params = None
    with open(html, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if TOTAL_MACS in line:
                i = line.index(TOTAL_MACS)
                macs = line[i + len(TOTAL_MACS):].split('(')[0].strip()
                if 'M' in macs:
                    macs = macs.split('M')[0]
                macs = int(macs)
                break
            if TOTAL_PARAMS in line:
                i = line.index(TOTAL_PARAMS)
                params = line[i + len(TOTAL_PARAMS):].split('(')[0].strip()
                params = int(params)
    return (macs, params)
