# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import sys
from archai.discrete_search.api import ArchaiModel
from archai.common.config import Config
from archai.discrete_search.evaluators.remote_azure_benchmark import RemoteAzureBenchmarkEvaluator
from aml.util.setup import configure_store


def reset_dlc(store, experiment_name, entity):
    """ Reset the qualcomm dlc files and associated metrics for the given entity."""
    changed = False
    name = entity['name']
    prefix = f'{experiment_name}/{name}'
    print(f"Resetting .dlc files for model {name}")
    store.delete_blobs(prefix, 'model.dlc')
    store.delete_blobs(prefix, 'model.quant.dlc')
    for k in ['mean', 'macs', 'params', 'stdev', 'total_inference_avg', 'error', 'f1_1k', 'f1_10k', 'f1_1k_f', 'f1_10k_f']:
        if k in entity:
            del entity[k]
            changed = True
    if changed:
        store.update_status_entity(entity)


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
    # change the metric key to the one used for Snapdragon F1 scoring
    target_config['metric_key'] = 'f1_1k'
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
        print(f"No 'complete' models found with required metric '{metric_key}'")
        sys.exit(1)

    # the RemoteAzureBenchmarkEvaluator only needs the archid actually, doesn't need the nn.Module.
    models = []
    for e in fully_trained:
        name = e['name']
        # if this has not been F1 scored yet then add it to our list.
        if 'benchmark_only' in e:
            models += [ArchaiModel(None, archid=name[3:])]
            # make sure we re-quantize the new fully trained model.
            reset_dlc(store, experiment_name, e)

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
