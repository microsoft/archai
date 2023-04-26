# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import sys
from archai.common.store import ArchaiStore


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def change_schema(store: ArchaiStore):
    """ Handy script for making batch changes to the azure table  """
    for e in store.get_all_status_entities():
        status = e['status']
        if status == 'preparing':
            e['status'] = 'complete'
            name = e['name']
            print(f'fixing {name}')
            store.merge_status_entity(e)


if __name__ == '__main__':
    experiment_name = os.getenv("EXPERIMENT_NAME", "facesynthetics")
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)
    change_schema(store)
