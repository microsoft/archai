# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import numpy as np
import sys
import tempfile
from pathlib import Path
from archai.discrete_search.api import ArchaiModel
from archai.discrete_search.search_spaces.config import ArchConfig
from archai.common.config import Config
from aml.util.pareto import calc_pareto_frontier
from search_space.hgnet import StackedHourglass
from aml.training.aml_training_evaluator import AmlPartialTrainingEvaluator
from aml.util.setup import configure_store


def main():
    # input and output arguments
    parser = argparse.ArgumentParser(description="Fully trains the final pareto curve models in parallel on Azure ML.")
    parser.add_argument("--config", type=str, help="location of the aml_search.yaml file", required=True)
    parser.add_argument("--output", type=str, help="location of local output files", default='output')
    parser.add_argument('--epochs', type=float, help='number of epochs to train (default 1)', default=30)
    parser.add_argument('--timeout', type=int, help='Timeout for training (in seconds)(default 28800 seconds = 8 hours)', default=28800)

    args = parser.parse_args()

    config = Config(args.config, resolve_env_vars=True)
    aml_config = config['aml']
    store = configure_store(aml_config)

    evaluator = AmlPartialTrainingEvaluator(config, args.output, args.epochs, args.timeout)
    store = evaluator.store

    experiment_name = aml_config['experiment_name']
    training = config['training']
    metric_key = training['metric_key']

    search_config = config['search']
    target_metric_key = search_config['target']['metric_key']

    ss_config = search_config['search_space']
    ss_config_params = ss_config.get('params', {})
    num_classes = ss_config_params.get('num_classes', 18)

    points = []
    for e in store.get_all_status_entities(status='complete'):
        if metric_key in e and target_metric_key in e:
            y = float(e[metric_key])
            x = float(e[target_metric_key])
            points += [[x, y, e]]

    if len(points) == 0:
        print(f"No models found with required metrics '{metric_key}' and '{target_metric_key}'")
        sys.exit(1)

    points = np.array(points)
    sorted = points[points[:, 0].argsort()]
    pareto = calc_pareto_frontier(sorted)
    print(f'Found {len(pareto)} models on pareto frontier')

    # change the key so the evaluator updates a different field this time and
    # does not think training is already complete.
    evaluator.metric_key = 'final_val_iou'
    training['metric_key'] = 'final_val_iou'

    models = []
    with tempfile.TemporaryDirectory() as tempdir:
        for i in pareto:
            x, y, e = sorted[i]
            id = e['name']
            iteration = int(e['iteration']) if 'iteration' in e else 0
            training_metric = y
            target_metric = x
            file_name = f'{id}.json'
            print(f'downloading {file_name} with {metric_key}={training_metric} and {target_metric_key}={target_metric} from iteration {iteration} ...')
            found = store.download(f'{experiment_name}/{id}', tempdir, specific_file=file_name)
            if len(found) == 1:
                arch_config = ArchConfig.from_file(os.path.join(tempdir, file_name))
                model = StackedHourglass(arch_config, num_classes=num_classes)
                models += [ArchaiModel(model, archid=id[3:], metadata={'config' : arch_config, 'entity': e})]
            else:
                print("Skipping model {id} because the .json arch config file was not found in the store.")

    # Ok, now fully train these models!
    print(f'Kicking off full training on {len(models)} models...')
    for model in models:
        e = model.metadata['entity']
        e = store.get_status(id)
        e['status'] = 'preparing'
        store.merge_status_entity(e)
        evaluator.send(model)
    evaluator.fetch_all()


if __name__ == "__main__":
    main()
