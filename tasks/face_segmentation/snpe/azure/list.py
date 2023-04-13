# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def list_models(con_str, experiment_name):
    parser = argparse.ArgumentParser(
        description="List all azure blob store assets.")
    parser.add_argument('--prefix', type=str, required=True, default=None,
                        help='List models matching this prefix')
    args = parser.parse_args()
    prefix = args.prefix

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)
    for blob in store.list_blobs(prefix):
        print(blob)


if __name__ == '__main__':
    experiment_name = os.getenv("EXPERIMENT_NAME", "facesynthetics")
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    list_models(con_str, experiment_name)
