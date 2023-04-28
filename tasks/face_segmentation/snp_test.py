# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import sys
from archai.discrete_search.api import ArchaiModel
from archai.common.config import Config
from archai.discrete_search.evaluators.remote_azure_benchmark import RemoteAzureBenchmarkEvaluator
from aml.util.setup import configure_store


def main():
    # input and output arguments
    parser = argparse.ArgumentParser(description="Runs Snapdragon F1 scoring on the final fully trained models produced by train_pareto.py.")
    parser.add_argument("--config", type=str, help="location of the aml_search.yaml file", required=True)

    args = parser.parse_args()

    config = Config(args.config, resolve_env_vars=True)
    aml_config = config['aml']
    experiment_name = aml_config['experiment_name']
    metric_key = 'final_val_iou'
    search_config = config['search']
    ss_config = search_config['search_space']
    ss_params = ss_config['params']
    in_channels = ss_params['in_channels']
    img_size = ss_params['img_size']
    target_config = search_config.get('target', {})
    target_name = target_config.pop('name', 'cpu')
    device_evaluator = None

    if target_name != 'snp':
        print(f"Snapdragon target is not configured in {args.config}")
        sys.exit(1)

    store = configure_store(aml_config)
    fully_trained = []
    for e in store.get_all_status_entities(status='complete'):
        if metric_key in e:
            fully_trained += [e]

    if len(fully_trained) == 0:
        print(f"No fully trained models found with required metric '{metric_key}'")
        sys.exit(1)

    # the RemoteAzureBenchmarkEvaluator only needs the archid actually, doesn't need the nn.Module.
    models = []
    for e in fully_trained:
        id = e['name']
        e['status'] = 'preparing'
        if 'benchmark_only' in e:
            del e['benchmark_only']
        store.update_status_entity(e)
        models += [ArchaiModel(None, archid=id[3:])]

    # kick off remote device training without the benchmark_only flag so we get the
    # F1 scores for these fully trained models.  Note the above results_path ensures the trained
    # models are uploaded back to our models blob store.
    input_shape = (1, in_channels, *img_size[::-1])
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
