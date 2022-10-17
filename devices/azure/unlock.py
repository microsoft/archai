# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from status import get_all_status_entities, merge_status_entity

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def unlock(node_name=None):
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    # fix up the 'date' property for uploaded blobs
    for e in get_all_status_entities():
        name = e['name']
        node = e['node'] if 'node' in e else None
        changed = False
        if 'node' in e:
            if node_name and node_name != node:
                continue
            e['node'] = ''
            changed = True

        if changed:
            print(f"Unlocking job {name} on node {node}")
            merge_status_entity(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unlock jobs for given node or unlock all jobs.')
    parser.add_argument('--node', help='Optional node name (default None).')
    args = parser.parse_args()
    unlock(args.node)
