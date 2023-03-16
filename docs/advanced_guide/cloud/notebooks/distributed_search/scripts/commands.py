# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml.entities import UserIdentityConfiguration


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
        environment=environment_name,
        identity=UserIdentityConfiguration(),
        command="""python3 train.py \
                --data_dir "${{inputs.data}}" \
                --output ${{outputs.results}} """ + args
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


def make_training_pipeline_command(description, hex_config, code_dir, compute_cluster_name, experiment_name,
                                   results_path, environment_name, training_epochs):

    fixed_args = f'--config "{hex_config}" ' + \
                f'--description "{description}" ' + \
                f'--compute_cluster_name "{compute_cluster_name}" ' + \
                f'--experiment_name "{experiment_name}" ' + \
                f'--environment_name "{environment_name}" ' + \
                f'--epochs "{training_epochs}" '

    return command(
        name="training",
        display_name="Distributed model training pipeline",
        description="Starts a separate pipeline to do distributed partial training of a given set of models.",
        inputs={
            "models": Input(type="uri_folder"),
            "data": Input(type="uri_folder")
        },
        outputs={
            "results": Output(type="uri_folder", path=results_path, mode="rw_mount")
        },
        code=code_dir,
        identity=UserIdentityConfiguration(),
        command="""python3 training_pipeline.py
            --models_path "${{inputs.models}}" \
            --datastore_path "${{inputs.data}}" \
            --results_path "${{outputs.results}}" """ + fixed_args,
        environment=environment_name,
    )
