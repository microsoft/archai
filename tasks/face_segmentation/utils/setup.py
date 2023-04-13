# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from glob import glob
from shutil import copyfile
from archai.common.config import Config
from archai.common.store import ArchaiStore
from azure.ai.ml.entities._credentials import AccountKeyConfiguration
from azure.ai.ml.entities import AzureBlobDatastore
import archai.common.azureml_helper as aml_helper


def configure_store(aml_config: Config, blob_container_name: str = None) -> ArchaiStore:
    con_str = aml_config.get('connection_str')
    if not con_str:
        print("Please set environment variable 'MODEL_STORAGE_CONNECTION_STRING' containing the Azure storage account connection " +
              "string for the Azure storage account you want to use to control this experiment.")
        sys.exit(1)

    if blob_container_name is None:
        blob_container_name = aml_config.get('blob_container_name', 'models')
    experiment_name = aml_config.get('experiment_name', 'facesynthetics')
    partition_key = aml_config.get('partition_key', 'main')
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    return ArchaiStore(storage_account_name, storage_account_key, blob_container_name, experiment_name, partition_key)


def register_datastore(ml_client, data_store_name, blob_container_name, storage_account_name, storage_account_key, experiment_name):
    try:
        credentials = AccountKeyConfiguration(account_key=storage_account_key)
        model_store = ml_client.datastores.get(data_store_name)
        if model_store.container_name != blob_container_name:
            raise Exception(f'The container name does not match. Only the credentials on {data_store_name} can be updated')
        if model_store.account_name != storage_account_name:
            raise Exception(f'The storage account name does not match. Only the credentials on {data_store_name} can be updated')
        model_store.credentials = credentials
    except:
        model_store = AzureBlobDatastore(
            name=data_store_name,
            description="Datastore pointing to a blob container.",
            account_name=storage_account_name,
            container_name=blob_container_name,
            credentials=credentials,
        )

    ml_client.create_or_update(model_store)
    return f'azureml://datastores/{data_store_name}/paths/{experiment_name}'


def create_cluster(ml_client, config, key):
    section = config[key]
    compute_name = section['name']
    size = section['size']
    location = section['location']
    max_instances = section.get('max_instances', 1)
    aml_helper.create_compute_cluster(ml_client, compute_name, size=size, location=location, max_instances=max_instances)
    return compute_name


def copy_code_folder(src_dir, target_dir):
    """ Copies the code folder into a separate folder.  This is needed otherwise the pipeline will fail with
    UserError: The code snapshot was modified in blob storage, which could indicate tampering.
    If this was unintended, you can create a new snapshot for the run. To do so, edit any
    content in the local source directory and resubmit the run.
    """
    os.makedirs(target_dir, exist_ok=True)
    for path in glob(os.path.join(src_dir, '*.py')):
        file = os.path.basename(path)
        print(f"copying source file : {file} to {target_dir}")
        copyfile(path, os.path.join(target_dir, file))
    for name in os.listdir(src_dir):
        path = os.path.join(src_dir, name)
        if os.path.isdir(path):
            copy_code_folder(path, os.path.join(target_dir, name))
