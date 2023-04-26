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
from archai.common.store import ArchaiStore
from shutil import copyfile
from archai.common.monitor import JobCompletionMonitor
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
    def send(self, arch: ArchaiModel, budget: Optional[float] = None) -> None:
        self.models += [arch.arch.get_archid()]

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        snapshot = self.models
        self.models = []  # reset for next run.

        training_type = 'partial' if self.partial_training else 'full'
        print(f"AmlTrainingValAccuracy: Starting {training_type} training on {len(snapshot)} models")

        # train all the models listed in the snapshot on a GPU cluster so we get much training
        # happening in parallel which greatly reduces the overall Archai Search process.
        description = "AmlTrainingValAccuracy partial training"
        pipeline_job, model_names = start_training_pipeline(
            description,  self.ml_client, self.store, snapshot,
            self.compute_cluster_name, self.datastore_path, self.models_path, self.local_output,
            self.experiment_name, self.environment_name, self.training_epochs, save_models=False)

        job_id = pipeline_job.name
        print(f'AmlTrainingValAccuracy: Started training pipeline: {job_id}')

        # wait for all the parallel training jobs to finish
        metric_key = 'vac_acc'
        keys = [metric_key]
        monitor = JobCompletionMonitor(self.store, self.ml_client, keys, job_id, self.timeout)
        results = monitor.wait(model_names)

        # save the results to the output folder (which is mapped by the AML pipeline to our
        # blob store under the container 'models' in the folder named the same as the
        # experiment_name)
        results_path = f'{self.local_output}/models.json'
        with open(results_path, 'w') as f:
            f.write(json.dumps(results, indent=2))

        # save the archai log also which can be handy for debugging later.
        log = 'archai.log'
        if os.path.isfile(log):
            copyfile(log, f'{self.local_output}/{log}')

        # extract the array of accuracies for our return value this is the metric that the
        # Archai search needs to figure out which models to continue to evolve and which are
        # not so good.
        accuracies = []
        for i, m in enumerate(results['models']):
            val_acc = m[metric_key]
            accuracies += [val_acc]

        print(f'AmlTrainingValAccuracy: fetch_all returning : {accuracies}')
        return accuracies
