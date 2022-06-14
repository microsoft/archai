# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import random
import sys
import platform
from azure.storage.blob import BlobClient, ContainerClient
from status import get_status, merge_status_entity, get_all_status_entities, get_utc_date
from reset import reset_metrics
from delete import delete_blobs


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def read_names():
    path = os.path.join(os.path.dirname(__file__), 'names.txt')
    return [x.strip() for x in open(path, 'r').readlines()]


def get_node_id():
    return platform.node()


def allocate_name():
    names = read_names()
    for entity in get_all_status_entities():
        for key in entity.keys():
            if key == "name":
                value = entity[key]
                if value in names:
                    i = names.index(value)
                    del names[i]

    if len(names) == 0:
        print("### We are out of unique names!")
        sys.exit(1)
    return random.choice(names)


def upload_blob(name, file, blob_name=None):
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    filename = os.path.basename(file)
    if blob_name:
        blob = f"{name}/{blob_name}"
    else:
        blob = f"{name}/{filename}"

    container_client = ContainerClient.from_connection_string(conn_string, container_name="models")
    try:
        if not container_client.exists():
            container_client.create_container()
    except Exception as e:
        print(f"ignoring exception {e}")
    blob_client = BlobClient.from_connection_string(conn_string, container_name="models", blob_name=blob)

    with open(file, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)


def upload(model, name, priority=None, benchmark_only=False):
    if not name:
        name = allocate_name()
        print(f"Model friendly name allocated: {name}")
    e = get_status(name)
    if 'node' in e:
        print(f"This model is locked by {e['node']}")
        return

    e['status'] = 'uploading'
    e['node'] = get_node_id()  # lock the row until upload complete
    merge_status_entity(e)
    upload_blob(name, model)
    # remove any cached dlc files since they need to be redone now.
    delete_blobs(name, 'model.dlc')
    delete_blobs(name, 'model.quant.dlc')

    # record status of new record.
    reset_metrics(e, True, True, True)
    e['status'] = 'new'
    e['model_date'] = get_utc_date()
    if priority:
        e['priority'] = priority
    if benchmark_only:
        e['benchmark_only'] = 1 if benchmark_only else 0

    merge_status_entity(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upload a model to azure blob store defined by your ' +
        f'{CONNECTION_NAME} environment variable.')
    parser.add_argument('model', help='Path to the .onnx file to upload to Azure')
    parser.add_argument('--name', help='Friendly name if you already have one allocated.')
    parser.add_argument('--priority', type=int, help='Optional priority override for this job. ' +
                        'Larger numbers mean lower priority')
    parser.add_argument('--benchmark_only', help='Run only the benchmark tests and skip all the F1 tests',
                        action="store_true")
    args = parser.parse_args()
    upload(args.model, args.name, priority=args.priority, benchmark_only=args.benchmark_only)
