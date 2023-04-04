# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import uuid
from archai.common.store import ArchaiStore


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'
USAGE_TABLE_NAME = 'USAGE_TABLE_NAME'

USAGE_TABLE = 'usage'
CONNECTION_STRING = ''


def get_all_usage_entities(store, name_filter=None):
    """ Get all usage entities with optional device name filter """

    table_client = store._get_table_client()

    entities = []
    query = "PartitionKey eq 'main'"
    if name_filter:
        query += f" and name eq '{name_filter}'"

    try:
        for e in table_client.query_entities(query_filter=query):
            entities += [e]

    except Exception as e:
        print(f"### error reading table: {e}")

    return entities


def add_usage(store, name, start, end):
    e = store.get_entity(str(uuid.uuid4()))
    e['name'] = name
    e['start'] = start
    e['end'] = end
    store.update_status_entity(e)
    return e


def usage(con_str):
    parser = argparse.ArgumentParser(
        description='Print usage in .csv format using ' +
        f'{CONNECTION_NAME} environment variable.')
    parser.add_argument('--device', help='Optional match for the name column (default None).')
    args = parser.parse_args()

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key, status_table_name=USAGE_TABLE)
    entities = get_all_usage_entities(store, args.device)
    store.print_entities(entities)


if __name__ == '__main__':
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    usage(con_str)
