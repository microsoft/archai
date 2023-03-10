import os
from typing import List, Optional, Union
from overrides import overrides
import uuid

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator
from azure.ai.ml import MLClient, command, Input, Output, dsl
from azure.ai.ml.entities import UserIdentityConfiguration

scripts_dir = os.path.dirname(os.path.abspath(__file__))


class AmlTrainingValAccuracy(AsyncModelEvaluator):
    def __init__(self, compute_cluster_name, environment_name, datastore_path, storage_account_key, storage_account_name, ml_client: MLClient, training_epochs: float = 1.0):
        self.training_epochs = training_epochs
        self.compute_cluster_name = compute_cluster_name
        self.environment_name = environment_name
        self.datastore_path = datastore_path
        self.storage_account_key = storage_account_key
        self.storage_account_name = storage_account_name
        self.models = []
        self.ml_client = ml_client
        self.model_names = []

    @overrides
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        self.models += [arch.arch.get_archid()]

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        print(f"Ok, doing partial training on {len(self.models)} models")

        def make_train_model_command(id, output_path, archid, training_epochs):
            return command(
                name=f'train {id}',
                display_name=f'train {id}',
                inputs={
                    "data": Input(type="uri_folder")
                },
                outputs={
                    "results": Output(type="uri_folder", path=output_path, mode="rw_mount")
                },

                # The source folder of the component
                code=scripts_dir,
                identity= UserIdentityConfiguration(),
                command="""python3 train.py \
                        --data_dir "${{inputs.data}}" \
                        --output ${{outputs.results}} """ + \
                        f"--name {id} "     + \
                        f"--storage_key {id} "     + \
                        f"--storage_account_name {id} "     + \
                        f"--model_params {archid} "     + \
                        f"--subscription {self.ml_client.subscription_id} " + \
                        f"--resource_group {self.ml_client.resource_group_name} " + \
                        f"--workspace {self.ml_client.workspace_name} " + \
                        f"--epochs {training_epochs} ",
                environment=self.environment_name,
            )

        @dsl.pipeline(
            compute=self.compute_cluster_name,
            description="mnist archai partial training",
        )
        def mnist_pipeline(
            data_input
        ):
            outputs = {}
            for archid in self.models:
                id = str(uuid.uuid4())
                self.model_names += [id]
                job_id = 'uuid_' + id.replace('-', '_')
                output_path = f'{self.models_path}/{id}/'
                train_job = make_train_model_command(job_id, output_path, archid, self.training_epochs)(
                    data=data_input
                )
                outputs[id] = train_job.outputs.results

            return outputs

        pipeline = mnist_pipeline(
            data_input=Input(type="uri_folder", path=self.datastore_path)
        )

        # submit the pipeline job
        pipeline_job = self.ml_client.jobs.create_or_update(
            pipeline,
            # Project's name
            experiment_name="mnist_partial_training",
        )

        # wait until the job completes
        self.ml_client.jobs.stream(pipeline_job.name)
