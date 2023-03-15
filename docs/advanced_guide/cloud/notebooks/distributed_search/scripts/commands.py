# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from azure.ai.ml import command
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import UserIdentityConfiguration
import json
import os
from typing import Dict
from mldesigner.dsl import dynamic
import uuid
from model import MyModel
from store import ArchaiStore
from archai.discrete_search.search_spaces.config import ArchConfig

TRAINING_TIMEOUT = 3600  # one hour should be enough!


def make_train_model_command(id, storage_account_key, storage_account_name, subscription_id, resource_group_name, workspace_name, output_path, archid, code_dir, training_epochs, environment_name):
    """ This is a parametrized command for training a given model architecture.  We will stamp these out to create a distributed training pipeline. """
    args = \
        f'--name {id} ' + \
        f'--storage_account_key "{storage_account_key}" ' + \
        f'--storage_account_name "{storage_account_name}" ' + \
        f'--model_params "{archid}" ' + \
        f'--subscription "{subscription_id}" ' + \
        f'--resource_group "{resource_group_name}" ' + \
        f'--workspace "{workspace_name}" ' + \
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


def make_dynamic_training_subgraph(results_path: str,
                                   environment_name : str,
                                   storage_account_key : str,
                                   storage_account_name : str,
                                   subscription_id : str,
                                   resource_group_name : str,
                                   workspace_name : str,
                                   hex_config: str,
                                   scripts_dir : str,
                                   full_epochs : float):
    """ Create a dynamic subgraph that does not populate full training jobs until we know what all the top models are after the search completes.
    The top_models_folder is provided as an input along with the prepared dataset.  It returns the validation accuracy results """

    @dynamic
    def dynamic_training_subgraph(
        top_models_folder: Input(type="uri_folder"),
        data: Input(type="uri_folder")
    ) -> Output(type="uri_file"):
        """This dynamic subgraph will kick off a training job for each model in the input_top_models file.

        :param top_models_folder: Location of teh pareto.json file.
        :param data: Dataset to use to train the models.
        """
        # Read list of silos from a json file input
        # Note: calling `pipeline_input.result()` inside @dynamic will return actual value of the input.
        # In this case, input_silos is an PipelineInput object, so we need to call `result()` to get the actual value.
        path = top_models_folder.result()
        pareto_file = os.path.join(path, 'pareto.json')
        with open(pareto_file) as f:
            top_models = json.load(f)

        store = ArchaiStore(storage_account_key, storage_account_name)

        model_ids = []
        for a in top_models['top_models']:
            if type(a) is dict and 'nb_layers' in a:
                model = MyModel(ArchConfig(a))
                archid = model.get_archid()
                model_id = 'id_' + str(uuid.uuid4()).replace('-', '_')
                model_ids += [model_id]
                output_path = f'{results_path}/{model_id}'
                train_job = make_train_model_command(storage_account_key, storage_account_name, subscription_id, resource_group_name, workspace_name, 'full_training', output_path, archid, scripts_dir, full_epochs, environment_name)(
                    data=data
                )
                e = store.get_status(model_id)
                e['job_id'] = train_job.name
                store.update_status_entity(e)

        monitor_node = make_monitor_command(hex_config, scripts_dir, environment_name, TRAINING_TIMEOUT)(
            model_ids=",".join(model_ids)
        )

        return {
            "output": monitor_node.outputs.results
        }

    return dynamic_training_subgraph