# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from typing import List, Optional, Union
from overrides import overrides
from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator
from azure.ai.ml import MLClient, command, Input, Output, dsl
from store import ArchaiStore
from shutil import copyfile
from monitor import JobCompletionMonitor
from training_pipeline import start_training_pipeline
from utils import copy_code_folder


class AmlTrainingValAccuracy(AsyncModelEvaluator):
    def __init__(self,
                 config,
                 compute_cluster_name,
                 environment_name,
                 datastore_path,
                 models_path,
                 local_output,
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
        self.local_output = local_output
        self.config = config
        self.experiment_name = experiment_name
        self.models = []
        self.save_models = save_models
        self.ml_client = ml_client
        self.timeout = timeout_seconds
        self.store = None

        storage_account_key = config['storage_account_key']
        storage_account_name = config['storage_account_name']
        self.store = ArchaiStore(storage_account_name, storage_account_key)

    @overrides
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        self.models += [arch.arch.get_archid()]

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        snapshot = self.models
        self.models = []  # reset for next run.

        training_type = 'partial' if self.partial_training else 'full'
        print(f"AmlTrainingValAccuracy: Starting {training_type} training on {len(snapshot)} models")

        pipeline_job, model_names = start_training_pipeline(
            "mnist archai partial training",
            self.ml_client, self.store, snapshot, self.compute_cluster_name, self.datastore_path,
            self.models_path, self.experiment_name, self.environment_name, self.training_epochs)

        job_id = pipeline_job.name
        print(f'AmlTrainingValAccuracy: Started training pipeline: {job_id}')

        # wait for the job to finish
        monitor = JobCompletionMonitor(self.store, self.ml_client, job_id, self.timeout)
        results = monitor.wait(model_names)

        results_path = f'{self.local_output}/models.json'
        with open(results_path, 'w') as f:
            f.write(json.dumps(results, indent=2))

        # save the archai log.
        log = 'archai.log'
        if os.path.isfile(log):
            copyfile(log, f'{self.local_output}/{log}')

        # extract the array of accuracies for our return value.
        accuracies = []
        for i, m in enumerate(results['models']):
            val_acc = m['val_acc']
            accuracies += [val_acc]

        print(f'AmlTrainingValAccuracy: fetch_all returning : {accuracies}')
        return accuracies
