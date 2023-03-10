import os
from typing import List, Optional, Union
from overrides import overrides
import uuid
import time
from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator
from azure.ai.ml import MLClient, command, Input, Output, dsl
from azure.ai.ml.entities import UserIdentityConfiguration
from store import ArchaiStore

scripts_dir = os.path.dirname(os.path.abspath(__file__))


class AmlTrainingValAccuracy(AsyncModelEvaluator):
    def __init__(self, compute_cluster_name, environment_name, datastore_path, models_path, storage_account_key, storage_account_name, ml_client: MLClient, training_epochs: float = 1.0):
        self.training_epochs = training_epochs
        self.compute_cluster_name = compute_cluster_name
        self.environment_name = environment_name
        self.datastore_path = datastore_path
        self.models_path = models_path
        self.storage_account_key = storage_account_key
        self.storage_account_name = storage_account_name
        self.models = []
        self.ml_client = ml_client
        self.model_names = []
        self.store = ArchaiStore(storage_account_name, storage_account_key)

    @overrides
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        self.models += [arch.arch.get_archid()]

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        print(f"Ok, doing partial training on {len(self.models)} models")

        def make_train_model_command(id, output_path, archid, training_epochs):
            args = \
                f'--name {id} '     + \
                f'--storage_key "{self.storage_account_key}" '     + \
                f'--storage_account_name "{self.storage_account_name}" '     + \
                f'--model_params "{archid}" '     + \
                f'--subscription "{self.ml_client.subscription_id}" ' + \
                f'--resource_group "{self.ml_client.resource_group_name}" ' + \
                f'--workspace "{self.ml_client.workspace_name}" ' + \
                f'--epochs "{training_epochs}" '
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
                code=scripts_dir,
                identity= UserIdentityConfiguration(),
                command="""python3 train.py \
                        --data_dir "${{inputs.data}}" \
                        --output ${{outputs.results}} """ + args,
                environment=self.environment_name,
            )

        @dsl.pipeline(
            compute=self.compute_cluster_name,
            description="mnist archai partial training",
        )
        def mnist_partial_training_pipeline(
            data_input
        ):
            outputs = {}
            for archid in self.models:
                job_id = 'id_' + str(uuid.uuid4()).replace('-', '_')
                self.model_names += [job_id]
                output_path = f'{self.models_path}/{job_id}/'
                train_job = make_train_model_command(job_id, output_path, archid, self.training_epochs)(
                    data=data_input
                )
                print("--------------------------------------------------------------------------------")
                print(train_job)
                outputs[job_id] = train_job.outputs.results

            return outputs

        pipeline = mnist_partial_training_pipeline(
            data_input=Input(type="uri_folder", path=self.datastore_path)
        )

        # submit the pipeline job
        pipeline_job = self.ml_client.jobs.create_or_update(
            pipeline,
            # Project's name
            experiment_name="mnist_partial_training",
        )

        # wait for the job to finish
        completed = {}
        waiting = list(self.model_names)
        while len(waiting) > 0:
            id = waiting[0]
            e = self.store.get_existing_status(id)
            if e is not None and 'status' in e and (e['status'] == 'trained' or e['status'] == 'failed'):
                del waiting[0]
                completed[id] = e
            else:

                status = self.ml_client.jobs.get(pipeline_job.name).status
                if status == 'Completed':
                    # ok, all jobs are done, which means if we still have waiting tasks then they failed to
                    # even start.
                    break

                time.sleep(20)
                continue

        # awesome - they all completed!
        results = []
        for id in self.model_names:
            e = completed[id] if id in completed else {}
            if 'val_acc' in e:
                val_acc = float(e['val_acc'])
                results += [val_acc]
            else:
                # this one failed so just return a zero accuracy
                results += [float(0)]

        return results