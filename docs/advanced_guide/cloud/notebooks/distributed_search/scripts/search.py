import argparse
import torch
import json
from archai.discrete_search.api import SearchObjectives
from archai.discrete_search.evaluators import AvgOnnxLatency, TorchFlops
from archai.discrete_search.evaluators import TorchNumParameters
from archai.discrete_search.algos import EvolutionParetoSearch
from cnn_search_space import CNNSearchSpace
from aml_training_evaluator import AmlTrainingValAccuracy
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
# from mldesigner.dsl import dynamic  # can we use this to dynamically create the training commands?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to prepared dataset")
    parser.add_argument("--environment", type=str, help="name of AML environment to run the partial training jobs in")
    parser.add_argument("--compute", type=str, help="name of AML compute to run the partial training jobs on")
    parser.add_argument('--config', type=str, help='bin hexed config json info for mlclient')
    parser.add_argument("--output_dir", type=str, help="path to output data")
    parser.add_argument("--local_output", type=str, help="optional path to local output data (default output_dir)")

    args = parser.parse_args()

    environment_name = args.environment
    compute_name = args.compute
    data_dir = args.data_dir
    output_dir = args.output_dir

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

    space = CNNSearchSpace()

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

    ds = ml_client.datastores.get('datasets')
    print(f"Successfully found dataset from workspace {workspace_name} in resource group {resource_group}")

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
        # TODO: add device parameter when this is merged into the main branch.
        # model_evaluator=AvgOnnxLatency(input_shape=(1, 1, 28, 28), num_trials=3, device='cpu'),
        model_evaluator=AvgOnnxLatency(input_shape=(1, 1, 28, 28), num_trials=3),
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
                                               ml_client=ml_client,
                                               training_epochs=1),
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
        init_num_models=10,
        seed=1234,
        save_pareto_model_weights=False  # we are doing distributed training!
    )

    algo.search()

    # TBD, ok now from pareto curve pick 5 models to do full training on.

if __name__ == "__main__":
    main()
