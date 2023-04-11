# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import sys
from archai.common.store import ArchaiStore


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def cleanup_stale_pods(store: ArchaiStore):
    """ This script looks for kubernetes pods that are no longer running (e.g. the pod may have run out of
    memory or may have been stopped for whatever reason) and cleans up the state in our status table to
    ensure the job doesn't get zombied, it will be picked up by the next available pod.  """

    SCRIPT_DIR = os.path.dirname(__file__)
    sys.path += [os.path.join(SCRIPT_DIR, '..', 'util')]
    from shell import Shell
    shell = Shell()
    podinfo = shell.run(os.getcwd(), "kubectl get pods -n snpe -o json", print_output=False)
    podinfo = json.loads(podinfo)
    running = []
    for row in podinfo['items']:
        name = row['metadata']['name']
        status = row['status']['phase']
        if status == 'Running':
            running += [name]
        print(name, status)

    # unlock rows that belong to non-existent kubernetes pods.
    for e in store.get_all_status_entities(status='completed', not_equal=True):
        name = e['name']
        if 'node' in e and e['node']:
            node = e['node']
            status = e['status'] if 'status' in e else 'none'
            print(f"Found lock by {node} with status {status}")
            if node.startswith('snpe-quantizer') and node not in running:
                print(f"Clearing lock on non-existant pod: {node}")
                del e['node']
                store.update_status_entity(e)


if __name__ == '__main__':
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    cleanup_stale_pods(store)
