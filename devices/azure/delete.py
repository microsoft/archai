# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from azure.storage.blob import ContainerClient
from status import delete_status

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def delete_blobs(friendly_name, specific_file=None):
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    container = ContainerClient.from_connection_string(conn_string, container_name="models")
    if not container.exists():
        return

    prefix = f'{friendly_name}/'
    for blob in container.list_blobs(name_starts_with=prefix):
        file_name = blob.name[len(prefix):]
        if specific_file and file_name != specific_file:
            continue

        print("Deleting blob: " + file_name)
        container.delete_blob(blob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delete a model from azure using its friendly name')
    parser.add_argument('name', help='The friendly name allocated by the upload script.')
    parser.add_argument('--file', help='Delete just the one file associated with the friendly name.')
    args = parser.parse_args()
    delete_blobs(args.name, args.file)
    if not args.file:
        delete_status(args.name)
