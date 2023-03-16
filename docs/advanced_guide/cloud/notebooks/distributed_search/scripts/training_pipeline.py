import argparse
import uuid
import json
import os
from store import ArchaiStore
from commands import make_train_model_command
from azure.ai.ml import Input, MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential
from archai.discrete_search.search_spaces.config import ArchConfig
from azure.ai.ml import dsl
from utils import copy_code_folder


def start_training_pipeline(description, ml_client, store, model_architectures, compute_cluster_name, datastore_path,
                            models_path, experiment_name, environment_name, training_epochs):
    print("Training models: {model_architectures}")
    code_dir = copy_code_folder()
    model_names = []
    for archid in model_architectures:
        model_id = 'id_' + str(uuid.uuid4()).replace('-', '_')
        model_names += [model_id]

    @dsl.pipeline(
        compute=compute_cluster_name,
        description=description,
    )
    def mnist_partial_training_pipeline(
        data_input
    ):
        outputs = {}
        for i, archid in enumerate(model_architectures):
            model_id = model_names[i]

            output_path = f'{models_path}/{model_id}'
            train_job = make_train_model_command(
                output_path, code_dir, environment_name, model_id,
                store.storage_account_name, store.storage_account_key,
                ml_client.subscription_id, ml_client.resource_group_name, ml_client.workspace_name,
                archid, training_epochs)(
                data=data_input
            )

            print(f'Launching training job for model {model_id}')
            e = store.get_status(model_id)
            nb_layers,  kernel_size, hidden_dim = eval(archid)
            e["nb_layers"] = nb_layers
            e["kernel_size"] = kernel_size
            e["hidden_dim"] = hidden_dim
            e['status'] = 'preparing'
            e['epochs'] = training_epochs
            store.update_status_entity(e)
            outputs[model_id] = train_job.outputs.results

        return outputs

    training_pipeline = mnist_partial_training_pipeline(
        data_input=Input(type="uri_folder", path=datastore_path)
    )

    # submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        training_pipeline,
        experiment_name=experiment_name,
    )

    return (pipeline_job, model_names)


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="optional bin hex encoded config.json file ")
    parser.add_argument("--description", type=str, help="the pipeline description")
    parser.add_argument("--models_path", help="Location of our models.json file.")
    parser.add_argument("--compute_cluster_name", help="name of compute cluster to use")
    parser.add_argument("--datastore_path", help="location of dataset")
    parser.add_argument("--results_path", help="location to store the trained models")
    parser.add_argument("--experiment_name", help="name of AML experiment")
    parser.add_argument("--environment_name", help="AML conda environment to use")
    parser.add_argument('--epochs', type=float, help='number of epochs to train', default=0.001)
    parser.add_argument("--output", type=str, help="place to write the results", default='output')

    args = parser.parse_args()

    path = args.models_path
    pareto_file = os.path.join(path, 'pareto.json')
    with open(pareto_file) as f:
        pareto_models = json.load(f)

    model_architectures = []
    for a in pareto_models:
        if type(a) is dict and 'nb_layers' in a:
            config = ArchConfig(a)
            nb_layers = config.pick("nb_layers")
            kernel_size = config.pick("kernel_size")
            hidden_dim = config.pick("hidden_dim")
            archid = f'({nb_layers}, {kernel_size}, {hidden_dim})'
            model_architectures += [archid]

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

    store = ArchaiStore(storage_account_name, storage_account_key)

    start_training_pipeline(args.description, ml_client, store, model_architectures,
                            args.compute_cluster_name, args.datastore_path, args.results_path, args.experiment_name,
                            args.environment_name, args.epochs)


if __name__ == "__main__":
    main()
