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
from aml.util.setup import configure_store


def main():
    # input and output arguments
    parser = argparse.ArgumentParser(description="Runs Snapdragon F1 scoring on the final fully trained models.")
    parser.add_argument("--config", type=str, help="location of the aml_search.yaml file", required=True)

    args = parser.parse_args()

    config = Config(args.config, resolve_env_vars=True)
    aml_config = config['aml']
    con_str = aml_config.get('connection_str', '$')
    if '$' in con_str:
        print("Please set environment variable MODEL_STORAGE_CONNECTION_STRING containing the Azure" +
              "storage account connection string for the Azure storage account you want to use to " +
              "control this experiment.")
        return 1

    experiment_name = aml_config['experiment_name']
    metric_key = 'final_val_iou'
    search_config = config['search']
    target_metric_key = search_config['target']['metric_key']
    ss_config = search_config['search_space']
    target_config = search_config.get('target', {})
    target_name = target_config.pop('name', 'cpu')
    device_evaluator = None

    if target_name != 'snp':
        print(f"Snapdragon target is not configured in {args.config}")
        sys.exit(1)

    store = configure_store(con_str)
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

    # the RemoteAzureBenchmarkEvaluator only needs the archid actually, doesn't need the nn.Module.
    models = []
    for i in pareto:
        x, y, id = sorted[i]
        e = store.get_status(id)
        e['status'] = 'preparing'
        store.merge_status_entity(e)
        models += [ArchaiModel(None, archid=id[3:])]

    # kick off remote device training without the benchmark_only flag so we get the
    # F1 scores for these fully trained models.  Note the above results_path ensures the trained
    # models are uploaded back to our models blob store.
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
