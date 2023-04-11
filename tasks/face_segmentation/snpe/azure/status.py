# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from archai.common.store import ArchaiStore


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def status(con_str):
    parser = argparse.ArgumentParser(description='Print status in .csv format')
    parser.add_argument('--status', help='Optional match for the status column (default None).')
    parser.add_argument('--name', help='Optional name of single status row to return (default None).')
    parser.add_argument('--not_equal', '-ne', help='Switch the match to a not-equal comparison.', action="store_true")
    parser.add_argument('--locked', help='Find entities that are locked by a node.', action="store_true")
    parser.add_argument('--cols', help='Comma separated list of columns to report (default is to print all)')
    args = parser.parse_args()
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    entities = store.get_all_status_entities(args.status, args.not_equal)
    if args.locked:
        entities = [e for e in entities if 'node' in e and e['node']]
    if args.name:
        entities = [e for e in entities if 'name' in e and e['name'] == args.name]

    columns = None
    if args.cols:
        columns = [x.strip() for x in args.cols.split(',')]
    store.print_entities(entities, columns)


if __name__ == '__main__':
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)
    status(con_str)
