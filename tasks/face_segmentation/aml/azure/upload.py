# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def upload(con_str, experiment_name, args):
    parser = argparse.ArgumentParser(description='Upload a named model (and optional accompanying files) to your ' +
                                     'azure blob store')
    parser.add_argument('name', help='Friendly name of the folder to put this in.')
    parser.add_argument('file', help='Path to the file to upload to Azure ' +
                        'or a folder to upload all files in that folder to the same azure blob folder.')
    parser.add_argument('--priority', type=int, help='Optional priority override for this job. ' +
                        'Larger numbers mean lower priority')
    parser.add_argument('--reset', help='Reset stats for the model if it exists already.', action="store_true")
    args = parser.parse_args(args)
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)
    store.upload(f'{experiment_name}/args.name', args.file, args.reset, priority=args.priority)


if __name__ == '__main__':
    experiment_name = os.getenv("EXPERIMENT_NAME", "facesynthetics")
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    upload(con_str, experiment_name)
