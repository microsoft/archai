# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import logging
from azure.storage.blob import ContainerClient
from status import get_all_status_entities

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def list_models(prefix):
    logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    logger.setLevel(logging.ERROR)
    container = ContainerClient.from_connection_string(conn_string, container_name="models", logger=logger,
                                                       logging_enable=False)
    if not container.exists():
        print("Container 'models' does not exist")
        return (False, None)

    for blob in container.list_blobs(name_starts_with=prefix):
        print(blob.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="List all azure blob store assets.")
    parser.add_argument('--prefix', type=str, default=None,
                        help='List models matching this prefix')
    args = parser.parse_args()

    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    list_models(args.prefix)
