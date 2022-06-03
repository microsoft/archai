import os
import sys
from status import get_all_status_entities, update_status_entity

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'

conn_string = os.getenv(CONNECTION_NAME)
if not conn_string:
    print(f"Please specify your {CONNECTION_NAME} environment variable.")
    sys.exit(1)


# fix up the 'date' property for uploaded blobs
for e in get_all_status_entities():
    name = e['name']
    changed = False
    if 'node' in e:
        del e['node']
        changed = True

    if changed:
        print(f"Unlocking job {name}")
        update_status_entity(e)
