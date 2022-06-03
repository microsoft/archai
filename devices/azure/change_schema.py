# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import sys
from status import get_all_status_entities, update_status_entity, get_status_table_service
from azure.storage.blob import ContainerClient

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'

conn_string = os.getenv(CONNECTION_NAME)
if not conn_string:
    print(f"Please specify your {CONNECTION_NAME} environment variable.")
    sys.exit(1)


container_client = ContainerClient.from_connection_string(conn_string, container_name="models")


def get_last_modified_date(e, blob_name):
    name = '{}/{}'.format(e['name'], blob_name)
    blob_client = container_client.get_blob_client(name)
    if blob_client.exists():
        return blob_client.get_blob_properties().last_modified
    return None


service = get_status_table_service()


# fix the 'complete' status...
for e in get_all_status_entities(service=service):

    name = e['name']
    changed = False
    if 'memory_usage' in e:
        del e['memory_usage']
        changed = True

    if 'completed_by' in e:
        del e['completed_by']
        changed = True

    if 'elapsed' in e:
        del e['elapsed']
        changed = True

    if 'benchmark_only' not in e or not e['benchmark_only']:
        if 'f1_1k' not in e or 'f1_onnx' not in e or 'f1_10k' not in e or 'f1_1k_f' not in e:
            if e['status'] == 'complete':
                e['status'] = 'incomplete'
    else:
        if 'total_inference_avg' not in e:
            if e['status'] == 'complete':
                e['status'] = 'incomplete'
                changed = True
        else:
            total_inference_avg = json.loads(e['total_inference_avg'])
            if len(total_inference_avg) < 5:
                if e['status'] == 'complete':
                    e['status'] = 'incomplete'
                    changed = True

    if changed:
        print(f"Updating row {name}")
        update_status_entity(e, service=service)
