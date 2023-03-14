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
import tempfile
from shutil import copyfile


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

    def make_train_model_command(self, id, output_path, archid, code_dir, training_epochs):
        args = \
            f'--name {id} ' + \
            f'--storage_account_key "{self.storage_account_key}" ' + \
            f'--storage_account_name "{self.storage_account_name}" ' + \
            f'--model_params "{archid}" ' + \
            f'--subscription "{self.ml_client.subscription_id}" ' + \
            f'--resource_group "{self.ml_client.resource_group_name}" ' + \
            f'--workspace "{self.ml_client.workspace_name}" ' + \
            f'--epochs "{training_epochs}" '
        if self.save_models:
            args += '--save_models '
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
            environment=self.environment_name,
            )

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        snapshot = self.models
        self.models = []  # reset for next run.
        self.job_names = []
        self.job_archids = {}

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
                job_id = 'id_' + str(uuid.uuid4()).replace('-', '_')
                self.job_names += [job_id]
                self.job_archids[job_id] = archid
                output_path = f'{self.models_path}/{job_id}'
                train_job = self.make_train_model_command(job_id, output_path, archid, code_dir, self.training_epochs)(
                    data=data_input
                )
                outputs[job_id] = train_job.outputs.results

            return outputs

        pipeline = mnist_partial_training_pipeline(
            data_input=Input(type="uri_folder", path=self.datastore_path)
        )

        # submit the pipeline job
        pipeline_job = self.ml_client.jobs.create_or_update(
            pipeline,
            # Project's name
            experiment_name=self.experiment_name,
        )

        # wait for the job to finish
        completed = {}
        waiting = list(self.job_names)
        start = time.time()
        failed = 0
        while len(waiting) > 0:
            for i in range(len(waiting) - 1, -1, -1):
                id = waiting[i]
                e = self.store.get_existing_status(id)
                if e is not None and 'status' in e and (e['status'] == 'trained' or e['status'] == 'failed'):
                    del waiting[i]
                    completed[id] = e
                    if e['status'] == 'failed':
                        error = e['error']
                        print(f'Training job {id} failed with error: {error}')
                        failed += 1

            status = self.ml_client.jobs.get(pipeline_job.name).status
            if status == 'Completed':
                # ok, all jobs are done, which means if we still have waiting tasks then they failed to
                # even start.
                break
            elif status == 'Failed':
                raise Exception('Partial Training Pipeline failed')

            if len(waiting) > 0:
                if time.time() > self.timeout + start:
                    break
                print("AmlTrainingValAccuracy: Waiting 20 seconds for partial training to complete...")
                time.sleep(20)

        # awesome - they all completed!
        if len(completed) == 0:
            if time.time() > self.timeout + start:
                raise Exception(f'Partial Training Pipeline timed out after {self.timeout} seconds')
            else:
                raise Exception('Partial Training Pipeline failed to start')

        if failed == len(completed):
            raise Exception('Partial Training Pipeline failed all jobs')

        results = []
        for id in self.job_names:
            e = completed[id] if id in completed else {}
            if 'val_acc' in e:
                val_acc = float(e['val_acc'])
                results += [val_acc]
                archid = self.job_archids[id]
                self.result_cache[archid] = val_acc
            else:
                # this one failed so just return a zero accuracy
                results += [float(0)]

        timespan = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
        print(f'AmlTrainingValAccuracy: Distributed training completed in {timespan} seconds')
        print(f'AmlTrainingValAccuracy: returning {len(results)} results: {results}')
        return results
