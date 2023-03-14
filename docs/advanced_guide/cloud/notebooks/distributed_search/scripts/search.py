import argparse
import torch
import json
import time
import os
from archai.discrete_search.api import SearchObjectives
from archai.discrete_search.evaluators import AvgOnnxLatency, TorchFlops
from archai.discrete_search.evaluators import TorchNumParameters
from archai.discrete_search.algos import EvolutionParetoSearch
from archai.discrete_search.search_spaces.config import ArchParamTree, ConfigSearchSpace, DiscreteChoice
from aml_training_evaluator import AmlTrainingValAccuracy
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from model import MyModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to prepared dataset")
    parser.add_argument("--environment", type=str, help="name of AML environment to run the partial training jobs in")
    parser.add_argument("--experiment", type=str, help="name of AML experiment to use")
    parser.add_argument("--compute", type=str, help="name of AML compute to run the partial training jobs on")
    parser.add_argument('--config', type=str, help='bin hexed config json info for MLClient')
    parser.add_argument("--output_dir", type=str, help="path to output data")
    parser.add_argument("--local_output", type=str, help="optional path to local output data (default output_dir)")
    parser.add_argument("--init_num_models", type=int, default=10, help="Number of initial models to evaluate")
    parser.add_argument("--partial_training_epochs", type=float, default=0.01, help="Number of epochs for partial training")
    parser.add_argument("--full_training_epochs", type=float, default=10, help="Number of epochs for final training")

    args = parser.parse_args()

    environment_name = args.environment
    experiment_name = args.experiment
    compute_name = args.compute
    data_dir = args.data_dir
    output_dir = args.output_dir
    init_num_models = args.init_num_models
    partial_training_epochs = args.partial_training_epochs
    full_training_epochs = args.full_training_epochs

    print("Starting search with: ")
    print(f"Environment: {environment_name}")
    print(f"Compute: {compute_name}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    identity = AzureMLOnBehalfOfCredential()
    if args.config:
        print("Using AzureMLOnBehalfOfCredential...")
        workspace_config = str(bytes.fromhex(args.config), encoding='utf-8')
        print(f"Config: {workspace_config}")
        config = json.loads(workspace_config)
    else:
        print("Using DefaultAzureCredential...")
        config_file = "../.azureml/config.json"
        print(f"Config: {config_file}")
        config = json.load(open(config_file, 'r'))
        identity = DefaultAzureCredential()

    # setup the ConfigSearchSpace from given ArchParamTree configuration.
    arch_param_tree = ArchParamTree({
        'nb_layers': DiscreteChoice(list(range(1, 13))),
        'kernel_size': DiscreteChoice([1, 3, 5, 7]),
        'hidden_dim': DiscreteChoice([16, 32, 64, 128])
    })

    space = ConfigSearchSpace(MyModel, arch_param_tree, mutation_prob=0.3)

    # Make sure we have permission to access the ml_client, this will be needed in the
    # AmlTrainingValAccuracy evaluator so it can create child pipelines.
    subscription = config['subscription_id']
    resource_group = config['resource_group']
    workspace_name = config['workspace_name']
    storage_account_key = config['storage_account_key']
    storage_account_name = config['storage_account_name']

    ml_client = MLClient(
        identity,
        subscription,
        resource_group,
        workspace_name
    )

    ml_client.datastores.get('datasets')
    print(f"Successfully found dataset from workspace {workspace_name} in resource group {resource_group}")

    # create our search objectives
    search_objectives = SearchObjectives()

    search_objectives.add_constraint(
        'Number of parameters',
        TorchNumParameters(),
        constraint=(0.0, 1e6)
    )

    search_objectives.add_objective(
        # Objective function name (will be used in plots and reports)
        name='ONNX Latency (ms)',
        # ModelEvaluator object that will be used to evaluate the model
        model_evaluator=AvgOnnxLatency(input_shape=(1, 1, 28, 28), num_trials=3, device='cpu'),
        # Optimization direction, `True` for maximization or `False` for minimization
        higher_is_better=False,
        # Whether this objective should be considered 'compute intensive' or not.
        compute_intensive=False
    )

    search_objectives.add_objective(
        name='FLOPs',
        model_evaluator=TorchFlops(torch.randn(1, 1, 28, 28)),
        higher_is_better=False,
        compute_intensive=False,
        # We may optionally add a constraint.
        # Architectures outside this range will be ignored by the search algorithm
        constraint=(0.0, 1e9)
    )

    search_objectives.add_objective(
        name='AmlTrainingValAccuracy',
        model_evaluator=AmlTrainingValAccuracy(compute_cluster_name=compute_name,
                                               environment_name=environment_name,  # AML environment name
                                               datastore_path=data_dir,  # AML datastore path
                                               models_path=output_dir,
                                               storage_account_key=storage_account_key,  # for ArchaiStore
                                               storage_account_name=storage_account_name,
                                               experiment_name=experiment_name,
                                               ml_client=ml_client,
                                               save_models=False,  # these partially trained models are not useful
                                               partial_training=True,
                                               training_epochs=partial_training_epochs),
        higher_is_better=True,
        compute_intensive=True
    )

    local_output = args.local_output
    if not local_output:
        local_output = args.output_dir

    algo = EvolutionParetoSearch(
        space,
        search_objectives,
        None,
        local_output,
        num_iters=5,
        init_num_models=init_num_models,
        seed=int(time.time()),
        save_pareto_model_weights=False  # we are doing distributed training!
    )

    results = algo.search()
    pareto = results.get_pareto_frontier()["models"]
    top_models = []
    for m in pareto:
        config = m.metadata['config']
        m.arch = MyModel(config)
        d = config.to_dict()
        id = str(m.archid)
        d['archid'] = id
        top_models += [d]

    with open(os.path.join(local_output, 'pareto.json'), 'w') as f:
        f.write(json.dumps(top_models, indent=2))

    print(f"Doing full training on {len(pareto)} best models")

    full_training = AmlTrainingValAccuracy(compute_cluster_name=compute_name,
                                           environment_name=environment_name,  # AML environment name
                                           datastore_path=data_dir,  # AML datastore path
                                           models_path=output_dir,
                                           storage_account_key=storage_account_key,  # for ArchaiStore
                                           storage_account_name=storage_account_name,
                                           experiment_name=experiment_name,
                                           ml_client=ml_client,
                                           save_models=True,
                                           partial_training=False,
                                           training_epochs=full_training_epochs)

    for m in pareto:
        full_training.send(m, None)

    # wait for all jobs to finish
    accuracies = full_training.fetch_all()

    # save name of top models
    print('Top model results: ')
    names = full_training.job_names
    for i, m in enumerate(pareto):
        name = names[i]
        val_acc = accuracies[i]
        d = top_models[i]
        d['val_acc'] = val_acc
        d['job_id'] = name

    results = {
        'init_num_models': init_num_models,
        'partial_training_epochs': partial_training_epochs,
        'full_training_epochs': full_training_epochs,
        'top_models': top_models
    }
    indented = json.dumps(results, indent=2)
    print(indented)
    with open(os.path.join(local_output, 'top_models.json'), 'w') as f:
        f.write(indented)


if __name__ == "__main__":
    main()
