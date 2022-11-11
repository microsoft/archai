# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import sys
from status import get_all_status_entities, update_status_entity, get_status_table_service, get_connection_string
from runner import has_model
from azure.storage.blob import ContainerClient

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def run():
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    container = ContainerClient.from_connection_string(conn_string, container_name="models")

    service = get_status_table_service(get_connection_string())

    # fix the 'complete' status...
    for e in get_all_status_entities(service=service):

        name = e['name']
        changed = False
        if 'quantized' not in e:
            is_quantized = container.get_blob_client(f'{name}/model.quant.dlc').exists()
            if is_quantized:
                e['quantized'] = True
                changed = True
                print(f"Setting quantized to True for {name}")
            elif 'quantized' in e:
                e['quantized'] = False
                changed = True
                print(f"Setting quantized to False for {name}")

        if changed:
            update_status_entity(e, service=service)

if __name__ == '__main__':
    run()