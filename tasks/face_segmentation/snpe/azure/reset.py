# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def reset(con_str, experiment_name):
    parser = argparse.ArgumentParser(
        description='Reset the named entity.')
    parser.add_argument('name', help='The friendly name to reset or "*" to reset all rows', default=None)
    args = parser.parse_args()
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)

    entities = []
    if args.name == "*":
        entities = [e for e in store.get_all_status_entities()]
    else:
        e = store.get_existing_status(args.name)
        if e is None:
            print(f"Entity {args.name} not found")
            sys.exit(1)
        else:
            entities = [e]

    for e in entities:
        name = e['name']
        store.reset(e['name'], ['benchmark_only', 'model_date'])
        store.delete_blobs(name, 'model.dlc')
        store.delete_blobs(name, 'model.quant.dlc')


if __name__ == '__main__':
    experiment_name = os.getenv("EXPERIMENT_NAME", "facesynthetics")
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    reset(con_str, experiment_name)
