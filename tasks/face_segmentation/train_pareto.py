# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import json
import os
import numpy as np
import sys
import tempfile
from pathlib import Path
from archai.common.store import ArchaiStore
from archai.discrete_search.api import ArchaiModel
from archai.discrete_search.search_spaces.config import ArchConfig
from archai.common.config import Config
from archai.common.monitor import JobCompletionMonitor
from aml.util.pareto import calc_pareto_frontier
from search_space.hgnet import StackedHourglass
from aml.util.setup import get_valid_arch_id
from aml.training.aml_training_evaluator import AmlPartialTrainingEvaluator
from search_space.hgnet import HgnetSegmentationSearchSpace
from archai.discrete_search.evaluators.remote_azure_benchmark import RemoteAzureBenchmarkEvaluator


def main():
    # input and output arguments
    parser = argparse.ArgumentParser(description="Fully trains the final pareto curve models in parallel on Azure ML.")
    parser.add_argument("--config", type=str, help="location of the aml_search.yaml file", required=True)
    parser.add_argument("--description", type=str, help="the pipeline description", default="fully train models")
    parser.add_argument('--epochs', type=float, help='number of epochs to train (default 1)', default=30)
    parser.add_argument('--timeout', type=int, help='Timeout for training (in seconds)(default 28800 seconds = 8 hours)', default=28800)

    args = parser.parse_args()

    config = Config(args.config, resolve_env_vars=True)
    aml_config = config['aml']
    con_str = aml_config.get('connection_str', '$')
    if '$' in con_str:
        print("Please set environment variable MODEL_STORAGE_CONNECTION_STRING containing the Azure" +
              "storage account connection string for the Azure storage account you want to use to " +
              "control this experiment.")
        return 1

    results_path = Path(aml_config['results_path'])
    evaluator = AmlPartialTrainingEvaluator(config, results_path, args.epochs, args.timeout)
    store = evaluator.store

    experiment_name = aml_config['experiment_name']
    training = config['training']
    metric_key = training['metric_key']

    search_config = config['search']
    target_metric_key = search_config['target']['metric_key']

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

    # change the key so the evaluator updates a different field this time and
    # does not thing training is already complete.
    evaluator.metric_key = 'final_val_iou'

    models = []
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
                models += [ArchaiModel(model, archid=id[3:], metadata={'config' : arch_config})]

    # Ok, now fully train these models!
    for model in models:
        evaluator.send(model)
    evaluator.fetch_all()

    # Let's kick off remote device training without the benchmark_only flag so we get the
    # F1 scores for these fully trained models.  Note the above results_path ensures the trained
    # models are uploaded back to our models blob store.
    ss_config = search_config['search_space']
    target_config = search_config.get('target', {})
    target_name = target_config.pop('name', 'cpu')
    device_evaluator = None

    if target_name == 'snp':

        search_space = HgnetSegmentationSearchSpace(
            seed=42,  # not important in this case.
            **ss_config.get('params', {}),
        )

        input_shape = (1, search_space.in_channels, *search_space.img_size[::-1])
        device_evaluator = RemoteAzureBenchmarkEvaluator(
                input_shape=input_shape,
                store=store,
                experiment_name=experiment_name,
                onnx_export_kwargs={'opset_version': 11},
                benchmark_only=0,  # do full F1 scoring this time.
                **target_config
            )

        for model in models:
            device_evaluator.send(model)
        device_evaluator.fetch_all()


if __name__ == "__main__":
    main()
