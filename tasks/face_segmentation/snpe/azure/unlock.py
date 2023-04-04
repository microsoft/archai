# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def unlock(con_str):
    parser = argparse.ArgumentParser(
        description='Unlock all jobs for given node or unlock all jobs.')
    parser.add_argument('--node', help='Optional node name (default None).')
    args = parser.parse_args()
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    store.unlock_all(args.node)


if __name__ == '__main__':
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    unlock(con_str)
