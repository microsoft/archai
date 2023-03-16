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


def start_training_pipeline(description, ml_client, store, model_architectures,
                            compute_cluster_name, datastore_uri, results_uri, output_folder,
                            experiment_name, environment_name, training_epochs, save_models):
    print(f"Training models: {model_architectures}")
    print(f"Cluster: {compute_cluster_name}")
    print(f"Dataset: {datastore_uri}")
    print(f"Output: {results_uri}")
    print(f"Env: {environment_name}")
    print(f"Epochs: {training_epochs}")

    code_dir = copy_code_folder()
    model_names = []
    for archid in model_architectures:
        model_id = 'id_' + str(uuid.uuid4()).replace('-', '_')
        model_names += [model_id]

    root_uri = results_uri
    i = root_uri.rfind('/')
    if i > 0:
        root_uri = root_uri[:i]

    # create new status rows and models.json for these new jobs.
    models = []
    for i, archid in enumerate(model_architectures):
        model_id = model_names[i]
        print(f'Launching training job for model {model_id}')
        e = store.get_status(model_id)
        nb_layers,  kernel_size, hidden_dim = eval(archid)
        e["nb_layers"] = nb_layers
        e["kernel_size"] = kernel_size
        e["hidden_dim"] = hidden_dim
        e['status'] = 'preparing'
        e['epochs'] = training_epochs
        store.update_status_entity(e)
        models += [{
            'id': model_id,
            'status': 'training',
            'nb_layers': nb_layers,
            'kernel_size': kernel_size,
            'hidden_dim': hidden_dim,
            'epochs': training_epochs,
            'val_acc': e['val_acc'] if 'val_acc' in e else 0.0
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
        for i, archid in enumerate(model_architectures):
            model_id = model_names[i]

            output_path = f'{root_uri}/{model_id}'
            train_job = make_train_model_command(
                output_path, code_dir, environment_name, model_id,
                store.storage_account_name, store.storage_account_key,
                ml_client.subscription_id, ml_client.resource_group_name, ml_client.workspace_name,
                archid, training_epochs, save_models)(
                data=data_input
            )

            outputs[model_id] = train_job.outputs.results

        return outputs

    training_pipeline = parallel_training_pipeline(
        data_input=Input(type="uri_folder", path=datastore_uri)
    )

    # submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        training_pipeline,
        experiment_name=experiment_name,
    )

    # Write the new list of pending models so that the make_monitor_command
    # knows what to wait for.
    print("Writing updated models.json: ")
    print(json.dumps(results, indent=2))
    results_path = f'{output_folder}/models.json'
    with open(results_path, 'w') as f:
        f.write(json.dumps(results, indent=2))

    return (pipeline_job, model_names)


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="optional bin hex encoded config.json file ")
    parser.add_argument("--description", type=str, help="the pipeline description")
    parser.add_argument("--models_path", help="Location of our models.json file.")
    parser.add_argument("--compute_cluster_name", help="name of compute cluster to use")
    parser.add_argument("--datastore_uri", help="location of dataset datastore")
    parser.add_argument("--results_uri", help="location to store the trained models")
    parser.add_argument("--output_path", help="location to store the list of new models")
    parser.add_argument("--experiment_name", help="name of AML experiment")
    parser.add_argument("--environment_name", help="AML conda environment to use")
    parser.add_argument('--epochs', type=float, help='number of epochs to train', default=0.001)
    parser.add_argument("--save_models", help="AML conda environment to use", action="store_true")

    args = parser.parse_args()

    path = args.models_path
    print(f"Reading pareto.json from {path}")
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
                            args.compute_cluster_name, args.datastore_uri, args.results_uri, args.output_path,
                            args.experiment_name, args.environment_name, args.epochs, args.save_models)


if __name__ == "__main__":
    main()
