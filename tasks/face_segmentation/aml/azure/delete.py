# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def delete(con_str):
    parser = argparse.ArgumentParser(description='Delete a model from azure using its friendly name')
    parser.add_argument('name', help='The friendly name allocated by the upload script.')
    parser.add_argument('--file', help='Delete just the one file associated with the friendly name.')
    args = parser.parse_args()

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)
    store.delete_blobs(args.name, args.file)
    if not args.file:
        store.delete_status(args.name)


if __name__ == '__main__':
    experiment_name = os.getenv("EXPERIMENT_NAME", "facesynthetics")
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    delete(con_str, experiment_name)
