# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import logging
from azure.storage.blob import ContainerClient
from status import get_all_status_entities

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def has_model(friendly_name, conn_string, specific_file):
    return get_model_blob(friendly_name, conn_string, specific_file).exists()


def get_model_blob(friendly_name, conn_string, specific_file):
    logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    logger.setLevel(logging.ERROR)
    container = ContainerClient.from_connection_string(conn_string, container_name="models", logger=logger,
                                                       logging_enable=False)
    if not container.exists():
        return (False, None)

    return container.get_blob_client(f'{friendly_name}/{specific_file}')


def download_model(friendly_name, folder, conn_string, specific_file=None, all_files=False, no_dlc=False):
    """ Download the model given friendly name to the given folder using the given connection string
    and return the local path to that file including the folder.  If an optional specific_file is
    given then it tries to find and download that file from Azure.  If you set all_files to true
    it will download all files associated with the friendly name.  """
    logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    logger.setLevel(logging.ERROR)
    container = ContainerClient.from_connection_string(conn_string, container_name="models", logger=logger,
                                                       logging_enable=False)
    if not container.exists():
        return (False, None)

    if not os.path.isdir(folder):
        os.makedirs(folder)
    model_found = False
    model_name = None
    local_file = None
    prefix = f'{friendly_name}/'
    supported = ['.onnx', '.dlc', '.pt', '.pd']
    if no_dlc:
        supported = ['.onnx', '.pt', '.pd']

    for blob in container.list_blobs(name_starts_with=prefix):
        model_name = blob.name[len(prefix):]
        parts = os.path.splitext(model_name)
        download = False
        if all_files:
            download = True
            local_file = os.path.join(folder, model_name)
        elif specific_file:
            if specific_file != model_name:
                continue
            else:
                download = True
                local_file = os.path.join(folder, model_name)

        elif len(parts) > 1:
            filename, ext = parts
            if ext in supported:
                if '.quant' in filename:
                    local_file = os.path.join(folder, 'model.quant' + ext)
                else:
                    local_file = os.path.join(folder, 'model' + ext)
                download = True

        if download:
            print(f"Downloading file: {model_name} to {local_file} ...")
            blob_client = container.get_blob_client(blob)
            with open(local_file, 'wb') as f:
                data = blob_client.download_blob()
                f.write(data.readall())
            model_found = True
            if not all_files:
                break

    return (model_found, model_name, local_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download assets from azure blob store using friendly name, " +
        f"using your your {CONNECTION_NAME} environment variable.")
    parser.add_argument('--name', help='Friendly name of model to download (if not provided it downloads them all')
    parser.add_argument('--file', help='The optional name of the files to download instead of getting them all.')
    args = parser.parse_args()

    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    friendly_name = args.name
    if not friendly_name:
        friendly_names = [e['name'] for e in get_all_status_entities()]
    else:
        friendly_names = [friendly_name]

    specific_file = args.file
    all_files = False if specific_file else True

    for friendly_name in friendly_names:
        found, model, file = download_model(friendly_name, friendly_name, conn_string, specific_file, all_files)
        if not found and specific_file:
            print(f"file {specific_file} not found")
