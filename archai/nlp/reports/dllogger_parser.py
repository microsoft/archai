# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provides functions and methods capable of parsing dllogger outputs.
"""

import json
from typing import Any, Dict, Tuple


def parse_json_dlogger_file(json_file: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Parses a pre-defined JSON dlogger file into
        training and testing dictionary-based logs.

    Args:
        json_file: Path to the JSON dlloger file.

    Returns:
        (Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]): Training and testing logs.

    """

    with open(json_file, 'r') as f:
        lines = f.readlines()
    
    # For training logs, we need to ignore first and last lines
    train_logs = [json.loads(line.replace('DLLL ', '')) for line in lines[1:-1]]
    
    # Iterates through all training logs and constructs a new dictionary
    # with corresponding keys as the steps
    train_logs_dict = {}
    for log in train_logs:
        step = log['step'][0]

        # There is a chance that given the same step, there are training and evaluation keys
        if step in train_logs_dict.keys():
            train_logs_dict[step].update(log['data'])
        else:
            train_logs_dict[step] = log['data']

        # Also adds missing step key to the recently created dictionary
        train_logs_dict[step]['step'] = step

    # Test logs will be present in the last line
    # We assert if a test-based key is present, otherwise the test logs will be invalid
    test_logs_dict = json.loads(lines[-1].replace('DLLL ', ''))['data']
    assert 'test_ppl' in test_logs_dict.keys(), f'There are test-related keys missing in {json_file}.'

    # Converts any potential lists into strings to avoid future crashes
    test_logs_dict.update((k, ','.join(map(str, v))) for k, v in test_logs_dict.items() if isinstance(v, list))

    return train_logs_dict, test_logs_dict
