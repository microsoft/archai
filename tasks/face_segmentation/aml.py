# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import yaml
from typing import Optional, Dict
from azure.ai.ml.entities import UserIdentityConfiguration
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml.entities._credentials import AccountKeyConfiguration
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml import dsl
from archai.common.config import Config
import archai.common.azureml_helper as aml_helper
from archai.common.store import ArchaiStore
fro archai.common.utils import copy_dir
from shutil import copyfile


confs_path = Path(__file__).absolute().parent / 'confs'


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


def data_prep_component(environment_name, datastore_path):
    return command(
        name="data_prep",
        display_name="Data preparation for training",
        description="Downloads the remote dataset to our blob store.",
        inputs={
            "name": Input(type='string')
        },
        outputs={
            "data": Output(type="uri_folder", path=datastore_path, mode="rw_mount")
        },

        # The source folder of the component
        code='data_prep',
        command="""python3 prep_data_store.py \
                --path ${{outputs.data}} \
                """,
        environment=environment_name,
    )


def search_component(environment_name, modelstore_path, output_path: Path, env_vars: Optional[Dict] = None):
    # we need a folder containing all the specific code we need here, which is not everything in this repo.
    scripts_path = output_path / 'scripts'
    if not scripts_path.exists():
        scripts_path.mkdir(parents=True)
    copyfile(str(confs_path / 'search.py'), str(scripts_path / 'search.py'))
    copy_dir(str(output_path / 'search_space'), str(scripts_path / 'search_space'))
    copy_dir(str(output_path / 'confs'), str(scripts_path / 'confs'))
    copy_dir(str(output_path / 'training'), str(scripts_path / 'training'))

    return command(
        name="search",
        display_name="Archai search job",
        description="Searches for the best face segmentation model.",
        inputs={
            "name": Input(type="uri_folder", mode="download")
        },
        outputs={
            "results": Output(type="uri_folder", path=modelstore_path, mode="rw_mount")
        },

        # The source folder of the component
        code=str(scripts_path),
        environment_variables=env_vars,
        command="""python3 search.py \
                --dataset_dir ${{inputs.data}} \
                --output_dir ${{outputs.results}} \
                --search_config confs/aml_search.yaml \
                """,
        environment=environment_name,
    )


def create_cluster(ml_client, config, key):
    section = config[key]
    compute_name = section['name']
    size = section['size']
    location = section['location']
    max_instances = section.get('max_instances', 1)
    aml_helper.create_compute_cluster(ml_client, compute_name, size=size, location=location, max_instances=max_instances)
    return compute_name


def main():
    parser = ArgumentParser("""This script runs the search in an Azure ML workspace.""")
    parser.add_argument('--output_dir', type=Path, help='Output directory for downloading results.', default='output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    seed = args.seed

    # Filters extra args that have the prefix `search_space`
    config_file = str(confs_path / 'aml_search.yaml')
    config = Config(config_file, resolve_env_vars=True)

    search_config = config['search']
    target_config = search_config.get('target', {})

    aml_config = config['aml']
    experiment_name = aml_config.pop('experiment_name', 'facesynthetics')

    con_str = target_config.pop('connection_str', None)
    if con_str is None or '$' in con_str:
        print("Please set environment variable {env_var_name} containing the Azure storage account connection " +
              "string for the Azure storage account you want to use to control this experiment.")
        return 1

    workspace_name = aml_config['workspace_name']
    subscription_id = aml_config['subscription_id']
    resource_group_name = aml_config['resource_group']
    env_vars = {
        'AZURE_SUBSCRIPTION_ID': subscription_id,
        'AML_RESOURCE_GROUP': resource_group_name,
        'AML_WORKSPACE_NAME': workspace_name,
    }

    # extract conda.yaml.
    with open('conda.yaml', 'w') as f:
        yaml.dump(aml_config['environment'].to_dict(), f)

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    print(f'Using workspace {workspace_name} and storage account: {storage_account_name}')

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name
    )
    print(f'Using workspace "{ml_client.workspace_name}" in resource group "{ml_client.resource_group_name}"')

    # Create aml computer clusters
    cpu_compute_name = create_cluster(ml_client, aml_config, 'search_cluster')
    gpu_compute_name = create_cluster(ml_client, aml_config, 'training_cluster')

    archai_job_env = aml_helper.create_environment_from_file(
        ml_client,
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
        conda_file="conda.yaml",
        version='1.0.3')
    environment_name = f"{archai_job_env.name}:{archai_job_env.version}"

    # Register the datastore with AML
    data_store_name = 'datasets'
    data_container_name = 'datasets'
    model_store_name = 'models'
    model_container_name = aml_config.get('blob_container_name', 'models')
    root_folder = experiment_name

    # make sure the datasets container exists
    store = ArchaiStore(storage_account_name, storage_account_key, blob_container_name=data_container_name, table_name=experiment_name)
    store.upload_blob(root_folder, config_file)

    # make sure the models container exists
    store = ArchaiStore(storage_account_name, storage_account_key, blob_container_name=model_container_name, table_name=experiment_name)
    store.upload_blob("config", config_file)

    results_path = register_datastore(ml_client, model_store_name, model_container_name, storage_account_name, storage_account_key, experiment_name)
    datastore_path = register_datastore(ml_client, data_store_name, data_container_name, storage_account_name, storage_account_key, experiment_name)

    @dsl.pipeline(
        compute=cpu_compute_name,
        description="FaceSynthetics Archai search pipeline",
    )
    def archai_search_pipeline():

        data_prep_job = data_prep_component(environment_name, datastore_path)(
            name=experiment_name
        )

        search_job = search_component(environment_name, results_path, output_dir, env_vars)(
            data=data_prep_job.outputs.data
        )

        return {
            "results": search_job.outputs.results
        }

    pipeline_job = ml_client.jobs.create_or_update(
        archai_search_pipeline(),
        # Project's name
        experiment_name=experiment_name,
    )

    import webbrowser
    webbrowser.open(pipeline_job.services["Studio"].endpoint)

    job_name = pipeline_job.name
    print(f'Started pipeline: {job_name}')

    return 0


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
