# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml.entities import UserIdentityConfiguration
import json
import os
from mldesigner.dsl import dynamic
import uuid
from store import ArchaiStore
from archai.discrete_search.search_spaces.config import ArchConfig


def make_train_model_command(output_path, code_dir, environment_name, id, storage_account_name, storage_account_key, subscription_id, resource_group_name, workspace_name, archid, training_epochs):
    """ This is a parametrized command for training a given model architecture.  We will stamp these out to create a distributed training pipeline. """

    args = f'--name "{id}" ' + \
        f'--storage_account_name "{storage_account_name}" ' + \
        f'--storage_account_key "{storage_account_key}" ' + \
        f'--subscription "{subscription_id}" ' + \
        f'--resource_group "{resource_group_name}" ' + \
        f'--workspace "{workspace_name}" ' + \
        f'--model_params "{archid}" ' + \
        f'--epochs "{training_epochs}" ' + \
        '--save_models '

    return command(
        name=f'train_{id}',
        display_name=f'train {id}',
        inputs={
            "data": Input(type="uri_folder")
        },
        outputs={
            "results": Output(type="uri_folder", path=output_path, mode="rw_mount")
        },

        # The source folder of the component
        code=code_dir,
        identity=UserIdentityConfiguration(),
        command="""python3 train.py \
                --data_dir "${{inputs.data}}" \
                --output ${{outputs.results}} """ + args,
        environment=environment_name,
    )


def make_monitor_command(hex_config, code_dir, environment_name, timeout=3600):
    """ This command waits up to some timeout for all the given training jobs to complete
     and returns the validation accuracy results """
    fixed_args = f'--config "{hex_config}" ' + \
                 f'--timeout {timeout} '
    return command(
        name="wait",
        display_name="Wait for training to complete",
        description="Waits for all distributed training jobs to complete.",
        inputs={
            "model_ids": Input(type="str")
        },
        outputs={
            "results": Output(type="uri_file")
        },
        code=code_dir,
        identity=UserIdentityConfiguration(),
        command="""python3 monitor.py
            --job_names "${{inputs.model_ids}}" \
            --results "${{outputs.results}} """ + fixed_args,
        environment=environment_name,
    )


def make_dynamic_training_subgraph(results_path, environment_name, storage_account_name, storage_account_key,
                                   subscription_id, resource_group_name, workspace_name,
                                   hex_config, scripts_dir, full_epochs, timeout=3600):
    """ Create a dynamic subgraph that does not populate full training jobs until we know what all the top models are after the search completes.
    The models_folder is provided as an input along with the prepared dataset.  It returns the validation accuracy results """
    args = \
        f'--results_path "{results_path}" ' + \
        f'--environment_name "{environment_name}" ' + \
        f'--storage_account_name "{storage_account_name}" ' + \
        f'--storage_account_key "{storage_account_key}" ' + \
        f'--subscription "{subscription_id}" ' + \
        f'--resource_group "{resource_group_name}" ' + \
        f'--workspace "{workspace_name}" ' + \
        f'--scripts_dir "{scripts_dir}" ' + \
        f'--full_epochs "{full_epochs}" '

    @dynamic
    def dynamic_training_subgraph(
        models_folder: Input(type="uri_folder"),
        data: Input(type="uri_folder")
    ) -> Output(type="uri_file"):
        """This dynamic subgraph will kick off a training job for each model in the models file.

        :param models_folder: Location of the pareto.json file.
        :param data: Dataset to use to train the models.
        """
        # Read list of silos from a json file input
        # Note: calling `pipeline_input.result()` inside @dynamic will return actual value of the input.
        # In this case, input_silos is an PipelineInput object, so we need to call `result()` to get the actual value.
        path = models_folder.result()
        pareto_file = os.path.join(path, 'pareto.json')
        with open(pareto_file) as f:
            pareto_models = json.load(f)

        store = ArchaiStore(storage_account_name, storage_account_key)

        model_ids = []
        for a in pareto_models:
            if type(a) is dict and 'nb_layers' in a:
                config = ArchConfig(a)
                nb_layers = config.pick("nb_layers")
                kernel_size = config.pick("kernel_size")
                hidden_dim = config.pick("hidden_dim")
                archid = f'({nb_layers}, {kernel_size}, {hidden_dim})'
                model_id = 'id_' + str(uuid.uuid4()).replace('-', '_')
                model_ids += [model_id]
                output_path = f'{results_path}/{model_id}'

                make_train_model_command(
                    output_path, scripts_dir, environment_name, model_id,
                    storage_account_name, storage_account_key, subscription_id, resource_group_name,
                    workspace_name, archid, full_epochs)(
                    data=data
                )
                e = store.get_status(model_id)
                e["nb_layers"] = nb_layers
                e["kernel_size"] = kernel_size
                e["hidden_dim"] = hidden_dim
                e['status'] = 'preparing'
                e['epochs'] = full_epochs
                store.update_status_entity(e)

        monitor_node = make_monitor_command(hex_config, scripts_dir, environment_name, timeout)(
            model_ids=",".join(model_ids)
        )

        return {
            "output": monitor_node.outputs.results
        }

    return dynamic_training_subgraph