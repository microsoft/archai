# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List, Optional, Union
from overrides import overrides
import uuid
from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator
from azure.ai.ml import MLClient, command, Input, Output, dsl
from store import ArchaiStore
from shutil import copyfile
from monitor import JobCompletionMonitor
from commands import make_train_model_command


class AmlTrainingValAccuracy(AsyncModelEvaluator):
    def __init__(self,
                 compute_cluster_name,
                 environment_name,
                 datastore_path,
                 models_path,
                 storage_account_key,
                 storage_account_name,
                 experiment_name,
                 ml_client: MLClient,
                 save_models: bool = True,
                 partial_training: bool = True,
                 training_epochs: float = 1.0,
                 timeout_seconds=3600):
        self.training_epochs = training_epochs
        self.partial_training = partial_training
        self.compute_cluster_name = compute_cluster_name
        self.environment_name = environment_name
        self.datastore_path = datastore_path
        self.models_path = models_path
        self.storage_account_key = storage_account_key
        self.storage_account_name = storage_account_name
        self.experiment_name = experiment_name
        self.models = []
        self.save_models = save_models
        self.ml_client = ml_client
        self.timeout = timeout_seconds
        self.result_cache = {}
        self.store = None
        self.store = ArchaiStore(storage_account_name, storage_account_key)

    def copy_code_folder(self):
        """ Copies the code folder into a separate folder.  This is needed otherwise the pipeline will fail with
        UserError: The code snapshot was modified in blob storage, which could indicate tampering.
        If this was unintended, you can create a new snapshot for the run. To do so, edit any
        content in the local source directory and resubmit the run.
        """
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        code_dir = 'temp_code'
        os.makedirs(code_dir, exist_ok=True)
        for file in os.listdir(scripts_dir):
            path = os.path.join(scripts_dir, file)
            if os.path.isfile(path):
                copyfile(path, os.path.join(code_dir, file))
        return code_dir

    @overrides
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        self.models += [arch.arch.get_archid()]

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        snapshot = self.models
        self.models = []  # reset for next run.
        self.model_names = []
        self.model_archs = {}

        training_type = 'partial' if self.partial_training else 'full'
        print(f"AmlTrainingValAccuracy: Starting {training_type} training on {len(snapshot)} models")

        code_dir = self.copy_code_folder()

        @dsl.pipeline(
            compute=self.compute_cluster_name,
            description="mnist archai partial training",
        )
        def mnist_partial_training_pipeline(
            data_input
        ):
            outputs = {}
            for archid in snapshot:
                if archid in self.result_cache:
                    print(f'### Already trained the model architecture {archid} ???')
                model_id = 'id_' + str(uuid.uuid4()).replace('-', '_')
                self.model_names += [model_id]
                self.model_archs[model_id] = archid
                output_path = f'{self.models_path}/{model_id}'
                train_job = make_train_model_command(
                    output_path, code_dir, self.environment_name,
                    self.store.storage_account_name, self.store.storage_account_key,
                    self.ml_client.subscription_id, self.ml_client.resource_group_name, self.ml_client.workspace_name,
                    model_id, archid, self.training_epochs)(
                    data=data_input
                )

                print('-------------------------------------------------------------------------------')
                print(train_job)
                print(f'Launching job {train_job.name} for model {model_id}')
                e = self.store.get_status(model_id)
                e['job_id'] = train_job.name
                self.store.update_status_entity(e)
                outputs[model_id] = train_job.outputs.results

            return outputs

        pipeline = mnist_partial_training_pipeline(
            data_input=Input(type="uri_folder", path=self.datastore_path)
        )

        # submit the pipeline job
        self.ml_client.jobs.create_or_update(
            pipeline,
            # Project's name
            experiment_name=self.experiment_name,
        )

        # wait for the job to finish
        monitor = JobCompletionMonitor(self.store, self.ml_client, self.timeout)
        results = monitor.wait(self.model_names)

        for i, val_acc in enumerate(results):
            id = self.model_names[i]
            archid = self.model_archs[id]
            self.result_cache[archid] = val_acc

        return results
