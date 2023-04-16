# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from pathlib import Path
from typing import List
from archai.common.store import ArchaiStore
from azure.ai.ml import MLClient
from archai.discrete_search.api import ArchaiModel
from archai.discrete_search.search_spaces.config import ArchConfig
from azure.ai.ml import command, Input, Output, dsl
from azure.ai.ml.entities import UserIdentityConfiguration
from archai.common.config import Config
from utils.setup import copy_code_folder, get_valid_arch_id
from shutil import copyfile
from archai.common.file_utils import TemporaryFiles


def training_component(output_path: str, code_dir: Path, config, training_epochs: int, metric_key: str, model_id: str, filename: str):
    # we need a folder containing all the specific code we need here, which is not everything in this repo.
    training = config['training']
    learning_rate = training['learning_rate']
    batch_size = training['batch_size']
    aml_config = config['aml']
    environment_name = aml_config['environment_name']

    fixed_args = f'--lr {learning_rate} --batch_size {batch_size} ' +\
                 f'--epochs {int(training_epochs)} --model_id {model_id} --metric_key {metric_key} ' +\
                 f'{filename}'

    return command(
        name="train",
        display_name="Archai training job",
        description="Trains a face segmentation model.",
        inputs={
            "data": Input(type="uri_folder", mode="download")
        },
        outputs={
            "results": Output(type="uri_folder", path=output_path, mode="rw_mount")
        },
        identity=UserIdentityConfiguration(),
        # The source folder of the component
        code=str(code_dir),
        command="""python3 train.py \
                --dataset_dir ${{inputs.data}} \
                --output_dir ${{outputs.results}} \
                """ + fixed_args,
        environment=environment_name,
    )


def start_training_pipeline(description: str, ml_client: MLClient, store: ArchaiStore,
                            model_architectures: List[ArchaiModel],
                            config: Config, training_epochs: int, output_folder: Path):
    """ Creates a new Azure ML Pipeline for training a set of models, updating the status of
    these jobs in a given Azure Storage Table.  This command does not wait for those jobs to
    finish.  For that use the monitor.py script which monitors the same Azure Storage Table
    to find out when the jobs have all finished.  The train.py script will update the table
    when each training job completes. """

    aml_config = config['aml']
    training_cluster = aml_config['training_cluster']
    compute_cluster_name = training_cluster['name']
    datastore_path = aml_config['datastore_path']
    root_uri = aml_config['results_path']
    environment_name = aml_config['environment_name']
    experiment_name = aml_config['experiment_name']
    metric_key = config['training'].get('metric_key', 'val_iou')

    print(f"Cluster: {compute_cluster_name}")
    print(f"Dataset: {datastore_path}")
    print(f"Output: {root_uri}")
    print(f"Environment: {environment_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Epochs: {training_epochs}")

    code_dir = Path('temp_code')
    os.makedirs(code_dir, exist_ok=True)
    config_dir = code_dir / 'confs'
    os.makedirs(config_dir, exist_ok=True)
    archs_dir = code_dir / 'archs'
    os.makedirs(archs_dir, exist_ok=True)
    copyfile('train.py', str(code_dir / 'train.py'))
    copy_code_folder('training', str(code_dir / 'training'))
    copy_code_folder('search_space', str(code_dir / 'search_space'))
    copy_code_folder('utils', str(code_dir / 'utils'))
    config.save(str(config_dir / 'aml_search.yaml'))

    models = []
    model_names = []
    for arch in model_architectures:

        model_id = get_valid_arch_id(arch)
        model_names += [model_id]
        print(f'Launching training job for model {model_id}')

        # upload the model architecture to our blob store so we can find it later.
        metadata: ArchConfig = arch.metadata['config']
        filename = str(archs_dir / f'{model_id}.json')
        metadata.to_file(filename)
        store.upload_blob(f'{experiment_name}/{model_id}', filename, blob_name=f'{model_id}.json')

        # create status entry in azure table
        e = store.get_status(model_id)
        e['experiment'] = experiment_name
        e['epochs'] = training_epochs
        e['status'] = 'preparing'
        store.merge_status_entity(e)
        models += [{
            'id': model_id,
            'status': 'training',
            'epochs': training_epochs,
            metric_key: e[metric_key] if metric_key in e else 0.0
        }]

    results = {
        'models': models
    }

    @dsl.pipeline(
        compute=compute_cluster_name,
        description=description,
    )
    def parallel_training_pipeline(
        data_input
    ):
        outputs = {}
        for arch in model_architectures:
            model_id = get_valid_arch_id(arch)
            output_path = f'{root_uri}/{model_id}'
            filename = f'archs/{model_id}.json'
            train_job = training_component(
                output_path, code_dir, config, training_epochs, metric_key, model_id, filename)(
                data=data_input
            )

            outputs[model_id] = train_job.outputs.results

        return outputs

    training_pipeline = parallel_training_pipeline(
        data_input=Input(type="uri_folder", path=datastore_path)
    )

    # submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        training_pipeline,
        experiment_name=experiment_name,
    )

    # Write the new list of pending models so that the make_monitor_command
    # knows what to wait for.
    print("Writing pending.json: ")
    print(json.dumps(results, indent=2))
    results_path = output_folder / 'pending.json'
    with open(results_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

    return (pipeline_job, model_names)
