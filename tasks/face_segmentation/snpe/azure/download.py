# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def download(con_str):
    parser = argparse.ArgumentParser(
        description="Download assets from azure blob store using friendly name.")
    parser.add_argument('--name', help='Friendly name of model to download (if not provided it downloads them all')
    parser.add_argument('--file', help='The optional name of the files to download instead of getting them all.')
    args = parser.parse_args()

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    friendly_name = args.name
    if not friendly_name:
        friendly_names = [e['name'] for e in store.get_all_status_entities()]
    else:
        friendly_names = [friendly_name]

    specific_file = args.file

    for friendly_name in friendly_names:
        downloaded = store.download(friendly_name, friendly_name, specific_file)
        if len(downloaded) == 0 and specific_file:
            print(f"file {specific_file} not found")


if __name__ == '__main__':
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    download(con_str)
