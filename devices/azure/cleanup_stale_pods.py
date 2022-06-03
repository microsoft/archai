import os
import json
import sys
from status import get_all_status_entities, update_status_entity

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'

conn_string = os.getenv(CONNECTION_NAME)
if not conn_string:
    print(f"Please specify your {CONNECTION_NAME} environment variable.")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(__file__)
sys.path += [os.path.join(SCRIPT_DIR, '..', 'util')]
from shell import Shell

shell = Shell()
podinfo = shell.run(os.getcwd(), "kubectl get pods -o json", print_output=False)
podinfo = json.loads(podinfo)
running = []
for row in podinfo['items']:
    name = row['metadata']['name']
    status = row['status']['phase']
    if status == 'Running':
        running += [name]
    print(name, status)

# unlock rows that belong to non-existant kubernetes pods.
for e in get_all_status_entities():
    name = e['name']
    changed = False
    if 'node' in e:
        node = e['node']
        status = e['status'] if 'status' in e else 'none'
        print(f"Found lock by {node} with status {status}")
        if node.startswith('snpe-quantizer') and node not in running:
            print(f"Clearing lock on non-existant pod: {node}")
            del e['node']
            changed = True
         
    if changed:
        print(f"Updating row {name}")
        update_status_entity(e)
