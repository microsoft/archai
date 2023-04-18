# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import yaml
from typing import Optional, Dict
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import UserIdentityConfiguration
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input, Output, dsl
from archai.common.config import Config
import archai.common.azureml_helper as aml_helper
from archai.common.store import ArchaiStore
from archai.common.file_utils import TemporaryFiles
from shutil import copyfile, rmtree
from aml.util.setup import register_datastore, configure_store, create_cluster, copy_code_folder


confs_path = Path(__file__).absolute().parent / 'confs'


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


def search_component(config, environment_name, seed, modelstore_path, output_path: Path):
    # we need a folder containing all the specific code we need here, which is not everything in this repo.
    scripts_path = output_path / 'scripts'
    os.makedirs(str(scripts_path), exist_ok=True)
    config_dir = scripts_path / 'confs'
    os.makedirs(str(config_dir), exist_ok=True)
    copyfile('search.py', str(scripts_path / 'search.py'))
    copyfile('train.py', str(scripts_path / 'train.py'))
    copy_code_folder('search_space', str(scripts_path / 'search_space'))
    copy_code_folder('training', str(scripts_path / 'training'))
    copy_code_folder(os.path.join('aml', 'training'), str(scripts_path / 'aml' / 'training'))
    copy_code_folder(os.path.join('aml', 'util'), str(scripts_path / 'aml' / 'util'))
    config.save(str(config_dir / 'aml_search.yaml'))

    aml_config = config['aml']
    timeout = int(aml_config.get('timeout', 3600))

    fixed_args = f'--seed {seed} --timeout {timeout} --search_config confs/aml_search.yaml'

    return command(
        name="search",
        display_name="Archai search job",
        description="Searches for the best face segmentation model.",
        is_deterministic=False,
        inputs={
            "data": Input(type="uri_folder")
        },
        outputs={
            "results": Output(type="uri_folder", path=modelstore_path, mode="rw_mount")
        },
        identity=UserIdentityConfiguration(),
        # The source folder of the component
        code=str(scripts_path),
        command="""python3 search.py \
                --dataset_dir ${{inputs.data}} \
                --output_dir ${{outputs.results}} \
                """ + fixed_args,
        environment=environment_name,
    )


def main(output_dir: Path, experiment_name: str, seed: int):
    if output_dir.exists():
        rmtree(str(output_dir))
    output_dir.mkdir(parents=True)

    # Filters extra args that have the prefix `search_space`
    config_file = str(confs_path / 'aml_search.yaml')
    config = Config(config_file, resolve_env_vars=True)

    aml_config = config['aml']
    con_str = aml_config.get('connection_str', '$')
    if '$' in con_str:
        print("Please set environment variable MODEL_STORAGE_CONNECTION_STRING containing the Azure" +
              "storage account connection string for the Azure storage account you want to use to " +
              "control this experiment.")
        return 1

    workspace_name = aml_config['workspace_name']
    subscription_id = aml_config['subscription_id']
    resource_group_name = aml_config['resource_group']

    # extract conda.yaml.
    with open('conda.yaml', 'w') as f:
        yaml.dump(aml_config['environment'].to_dict(), f)

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    print(f'Using storage account: {storage_account_name}')

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name
    )
    print(f'Using workspace "{ml_client.workspace_name}" in resource group "{ml_client.resource_group_name}"')

    # Create aml computer clusters
    cpu_compute_name = create_cluster(ml_client, aml_config, 'search_cluster')
    create_cluster(ml_client, aml_config, 'training_cluster')

    archai_job_env = aml_helper.create_environment_from_file(
        ml_client,
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
        conda_file="conda.yaml",
        version='1.0.13')
    environment_name = f"{archai_job_env.name}:{archai_job_env.version}"

    # Register the datastore with AML
    data_store_name = 'datasets'
    data_container_name = 'datasets'
    model_store_name = 'models'
    model_container_name = aml_config.get('blob_container_name', 'models')

    # register our azure datastores
    results_path = register_datastore(ml_client, model_store_name, model_container_name, storage_account_name, storage_account_key, experiment_name)
    datastore_path = register_datastore(ml_client, data_store_name, data_container_name, storage_account_name, storage_account_key, experiment_name)

    # save this in the output folder so it can be found by pipeline components.
    aml_config['experiment_name'] = experiment_name
    aml_config['environment_name'] = environment_name
    aml_config['datastore_path'] = datastore_path
    aml_config['results_path'] = results_path

    # make sure the datasets container exists
    store = configure_store(aml_config, data_container_name)

    # make sure the models container exists
    store = configure_store(aml_config, model_container_name)
    with TemporaryFiles() as tmp_files:
        filename = tmp_files.get_temp_file()
        config.save(filename)
        store.upload_blob(f"{experiment_name}/config", filename, 'aml_search.yaml')

    @dsl.pipeline(
        compute=cpu_compute_name,
        description="FaceSynthetics Archai search pipeline",
    )
    def archai_search_pipeline():

        data_prep_job = data_prep_component(environment_name, datastore_path)(
            name=experiment_name
        )

        search_job = search_component(config, environment_name, seed, results_path, output_dir)(
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
    parser = ArgumentParser("""This script runs the search in an Azure ML workspace.""")
    parser.add_argument('--output_dir', type=Path, help='Output directory for downloading results.', default='output')
    parser.add_argument('--experiment_name', default='facesynthetics')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)

    args = parser.parse_args()
    rc = main(args.output_dir, args.experiment_name, args.seed)
    sys.exit(rc)
