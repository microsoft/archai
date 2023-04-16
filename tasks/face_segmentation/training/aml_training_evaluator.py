# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from pathlib import Path
from typing import List, Optional, Union
from overrides import overrides
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator
from archai.common.config import Config
from shutil import copyfile
from archai.common.monitor import JobCompletionMonitor
from training.training_pipeline import start_training_pipeline
from azure.identity import DefaultAzureCredential
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.ml import MLClient
from utils.setup import configure_store


class AmlPartialTrainingValIOU(AsyncModelEvaluator):
    """ The AmlPartialTrainingValIOU evaluator launches partial training jobs"""
    def __init__(self,
                 config : Config,
                 local_output: Path,
                 tr_epochs: int = 1,
                 timeout_seconds=3600):
        self.config = config
        self.tr_epochs = int(tr_epochs)

        aml_config = config['aml']
        workspace_name = aml_config['workspace_name']
        subscription_id = aml_config['subscription_id']
        resource_group_name = aml_config['resource_group']

        identity = DefaultAzureCredential()
        if os.getenv('AZUREML_ROOT_RUN_ID'):
            identity = AzureMLOnBehalfOfCredential()

        self.ml_client = MLClient(
            credential=identity,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name
        )
        self.local_output = local_output
        self.models = []
        self.timeout = timeout_seconds
        self.store = configure_store(aml_config)

    @overrides
    def send(self, arch: ArchaiModel, budget: Optional[float] = None) -> None:
        self.models += [arch]

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        snapshot = self.models
        self.models = []  # reset for next run.

        print(f"AmlPartialTrainingValIOU: Starting training on {len(snapshot)} models")

        # train all the models listed in the snapshot on a GPU cluster so we get much training
        # happening in parallel which greatly reduces the overall Archai Search process.
        description = f"AmlPartialTrainingValIOU training {self.tr_epochs} epochs"
        pipeline_job, model_names = start_training_pipeline(
            description,  self.ml_client, self.store, snapshot, self.config, self.tr_epochs, self.local_output)

        job_id = pipeline_job.name
        print(f'AmlPartialTrainingValIOU: Started training pipeline: {job_id}')

        # wait for all the parallel training jobs to finish
        monitor = JobCompletionMonitor(self.store, self.ml_client, job_id, self.timeout)
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

        # extract the array of results for our return value this is the metric that the
        # Archai search needs to figure out which models to continue to evolve and which are
        # not so good.
        results = []
        for i, m in enumerate(results['models']):
            metric = m[self.metric_name]
            results += [metric]

        print(f'AmlPartialTrainingValIOU: fetch_all returning : {results}')
        return results
