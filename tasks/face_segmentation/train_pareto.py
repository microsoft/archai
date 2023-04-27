# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import json
import os
import numpy as np
import sys
import tempfile
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential
from archai.common.store import ArchaiStore
from archai.discrete_search.api import ArchaiModel
from archai.discrete_search.search_spaces.config import ArchConfig
from archai.common.config import Config
from archai.common.monitor import JobCompletionMonitor
from aml.util.pareto import calc_pareto_frontier
from search_space.hgnet import StackedHourglass
from aml.training.training_pipeline import start_training_pipeline
from aml.util.setup import get_valid_arch_id


def main():
    # input and output arguments
    parser = argparse.ArgumentParser(description="Fully trains the final pareto curve models in parallel on Azure ML.")
    parser.add_argument("--config", type=str, help="location of the aml_search.yaml file")
    parser.add_argument("--description", type=str, help="the pipeline description", default="train models")
    parser.add_argument("--output_dir", type=Path, help="location to store the list of pending models (pending.json)")
    parser.add_argument('--epochs', type=float, help='number of epochs to train (default 1)', default=30)
    parser.add_argument('--timeout', type=int, help='Timeout for training (in seconds)(default 28800 - 8 hours)', default=28800)

    args = parser.parse_args()

    config = Config(args.config, resolve_env_vars=True)
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
    experiment_name = aml_config['experiment_name']

    training = config['training']
    metric_key = training['metric_key']

    search = config['search']
    target_metric_key = search['target']['metric_key']

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    print(f'Using storage account: {storage_account_name}')
    store = ArchaiStore(storage_account_name, storage_account_key, table_name=experiment_name)
    points = []
    for e in store.get_all_status_entities(status='complete'):
        if metric_key in e and target_metric_key in e:
            y = float(e[metric_key])
            x = float(e[target_metric_key])
            id = e['name']
            points += [[x, y, id]]

    if len(points) == 0:
        print(f"No models found with required metrics '{metric_key}' and '{target_metric_key}'")
        sys.exit(1)

    points = np.array(points)
    sorted = points[points[:, 0].argsort()]
    pareto = calc_pareto_frontier(sorted)

    if os.getenv('AZUREML_ROOT_RUN_ID'):
        identity = AzureMLOnBehalfOfCredential()
    else:
        identity = DefaultAzureCredential()

    ml_client = MLClient(
        identity,
        subscription_id,
        resource_group_name,
        workspace_name
    )

    model_architectures = []
    with tempfile.TemporaryDirectory() as tempdir:
        for i in pareto:
            x, y, id = sorted[i]
            e = store.get_status(id)
            e['status'] = 'preparing'
            store.merge_status_entity(e)
            iteration = int(e['iteration']) if 'iteration' in e else 0
            training_metric = float(e[metric_key]) if metric_key in e else 0
            target_metric = float(e[target_metric_key]) if target_metric_key in e else 0
            file_name = f'{id}.json'
            print(f'downloading {file_name} with {metric_key}={training_metric} and {target_metric_key}={target_metric} from iteration {iteration} ...')
            found = store.download(f'{experiment_name}/{id}', tempdir, specific_file=file_name)
            if len(found) == 1:
                arch_config = ArchConfig.from_file(os.path.join(tempdir, file_name))
                model = StackedHourglass(arch_config, num_classes=18)
                model_architectures += [ArchaiModel(model, archid=id[3:], metadata={'config' : arch_config})]

    pipeline_job, model_names = start_training_pipeline(args.description, ml_client, store,
                                                        model_architectures, config, args.epochs, args.output_dir)

    job_id = pipeline_job.name
    print(f'train_pareto: Started training pipeline: {job_id}')

    # wait for all the parallel training jobs to finish
    keys = [metric_key]
    monitor = JobCompletionMonitor(store, ml_client, keys, job_id, args.timeout, throw_on_failure_rate=0.25)
    models = monitor.wait(model_names)['models']

    index = {}
    for m in models:
        id = m['id']
        index[id] = m

    # now reassemble all results in the right order (order of the sorted pareto curve)
    models = []
    for arch in model_architectures:
        model_id = get_valid_arch_id(arch)
        result = index[model_id]
        models += [result]

    results = {'models': models}

    # save the results to the output folder
    results_path = f'{args.output_dir}/models.json'
    summary = json.dumps(results, indent=2)
    with open(results_path, 'w') as f:
        f.write(summary)


if __name__ == "__main__":
    main()
