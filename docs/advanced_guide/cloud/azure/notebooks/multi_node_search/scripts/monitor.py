# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import argparse
import json
import os
from typing import List, Dict
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential
from archai.common.store import ArchaiStore
from azure.ai.ml import MLClient
from azure.ai.ml import dsl


class JobCompletionMonitor:
    """ This helper class uses the ArchaiStore to monitor the status of some long running
    training operations and the status of the Azure ML pipeline those jobs are running in
    and waits for them to finish (either successfully or with a failure)"""
    def __init__(self, store : ArchaiStore, ml_client : MLClient, pipeline_id=None, timeout=3600):
        """
        Initialize a JobCompletionMonitor instance.
        :param store: an instance of ArchaiStore to monitor the status of some long running training operations
        :param ml_client: an instance of MLClient to check the status of the Azure ML pipeline those jobs are running in
        :param pipeline_id: (optional) the ID of the Azure ML pipeline to monitor, if not provided we can get this from the ArchaiStore.
        :param timeout: (optional) the timeout in seconds
        """
        self.store = store
        self.ml_client = ml_client
        self.timeout = timeout
        self.pipeline_id = pipeline_id

    def wait(self, model_ids: List[str]) -> List[Dict[str, str]]:
        """
        Wait for all the training jobs to finish and return a list of dictionaries
        containing details about each model, including their training validation accuracies.
        :param model_ids: a list of training job IDs
        :return: a list of dictionaries containing details about each model
        """
        completed = {}
        waiting = list(model_ids)
        start = time.time()
        failed = 0

        while len(waiting) > 0:
            for i in range(len(waiting) - 1, -1, -1):
                id = waiting[i]
                e = self.store.get_status(id)
                if self.pipeline_id is None and 'pipeline_id' in e:
                    self.pipeline_id = e['pipeline_id']
                if e is not None and 'status' in e and (e['status'] == 'completed' or e['status'] == 'failed'):
                    del waiting[i]
                    completed[id] = e
                    if e['status'] == 'failed':
                        error = e['error']
                        print(f'Training job {id} failed with error: {error}')
                        failed += 1
                        self.store.update_status_entity(e)
                    else:
                        msg = f"{e['val_acc']}" if 'val_acc' in e else ''
                        print(f'Training job {id} completed with validation accuracy: {msg}')

            if len(waiting) == 0:
                break

            # check the overall pipeline status just in case training jobs failed to even start.
            pipeline_status = None
            try:
                if self.pipeline_id is not None:
                    train_job = self.ml_client.jobs.get(self.pipeline_id)
                    if train_job is not None:
                        pipeline_status = train_job.status
            except Exception as e:
                print(f'Error getting pipeline status for pipeline {self.pipeline_id}: {e}')

            if pipeline_status is not None:
                if pipeline_status == 'Completed':
                    # ok, all jobs are done, which means if we still have waiting tasks then they failed to
                    # even start.
                    break
                elif pipeline_status == 'Failed':
                    for id in waiting:
                        e = self.store.get_status(id)
                        if 'error' not in e:
                            e['error'] = 'Pipeline failed'
                        if 'status' not in e or e['status'] != 'completed':
                            e['status'] = failed
                        self.store.update_status_entity(e)
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

        # stitch together the models.json file from our status table.
        print('Top model results: ')
        models = []
        for id in model_ids:
            row = {'id': id}
            e = completed[id] if id in completed else {}
            for key in ['nb_layers', 'kernel_size', 'hidden_dim', 'val_acc', 'job_id', 'status', 'error', 'epochs']:
                if key in e:
                    row[key] = e[key]
            models += [row]

        results = {
            'models': models
        }

        timespan = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
        print(f'Training: Distributed training completed in {timespan} ')
        print(f'Training: returning {len(results)} results:')
        print(json.dumps(results, indent=2))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='bin hexed config json info for MLClient')
    parser.add_argument('--timeout', type=int, help='pipeline timeout in seconds (default 1 hour)', default=3600)
    parser.add_argument('--model_path', required=True, help='mounted path containing the pending.json file')
    parser.add_argument('--output', required=True, help='folder to write the results to)')

    args = parser.parse_args()
    output = args.output
    timeout = args.timeout
    model_path = args.model_path

    print(f"Monitor running with model_path={model_path}")
    if not os.path.isdir(model_path):
        raise Exception("### directory not found")

    models_file = os.path.join(model_path, 'pending.json')
    if not os.path.isfile(models_file):
        raise Exception("### 'pending.json' not found in --model_path")

    models = json.load(open(models_file))
    model_ids = [m['id'] for m in models['models']]

    identity = AzureMLOnBehalfOfCredential()
    if args.config:
        print("Using AzureMLOnBehalfOfCredential...")
        workspace_config = str(bytes.fromhex(args.config), encoding='utf-8')
        print(f"Config: {workspace_config}")
        config = json.loads(workspace_config)
    else:
        print("Using DefaultAzureCredential...")
        config_file = "../.azureml/config.json"
        print(f"Config: {config_file}")
        config = json.load(open(config_file, 'r'))
        identity = DefaultAzureCredential()

    subscription = config['subscription_id']
    resource_group = config['resource_group']
    workspace_name = config['workspace_name']
    storage_account_key = config['storage_account_key']
    storage_account_name = config['storage_account_name']

    ml_client = MLClient(
        identity,
        subscription,
        resource_group,
        workspace_name
    )

    store = ArchaiStore(storage_account_name, storage_account_key)

    monitor = JobCompletionMonitor(store, ml_client, timeout=timeout)
    results = monitor.wait(model_ids)
    if output is not None:
        # save the results with updated validation accuracies to models.json
        with open(os.path.join(output, 'models.json'), 'w') as f:
            f.write(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
