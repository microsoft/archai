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


class JobCompletionMonitor:
    """ This helper class uses the ArchaiStore to monitor the status of some long running
    training operations and the status of the Azure ML pipeline those jobs are running in
    and waits for them to finish (either successfully or with a failure)"""
    def __init__(self, store : ArchaiStore, ml_client : MLClient, metric_keys: List[str], pipeline_id=None, timeout=3600, throw_on_failure_rate=0.1):
        """
        Initialize a JobCompletionMonitor instance.
        :param store: an instance of ArchaiStore to monitor the status of some long running training operations
        :param ml_client: an instance of MLClient to check the status of the Azure ML pipeline those jobs are running in
        :param metric_keys: a list of column names to monitor and return from the Azure table.
        :param pipeline_id: (optional) the ID of the Azure ML pipeline to monitor, if not provided we can get this from the ArchaiStore.
        :param timeout: (optional) the timeout in seconds
        :param throw_on_failure_rate: (optional) what ratio of jobs failed (between 0 and 1) should result in raising an an exception.
        Zero means throw exception on any failure.
        This is handy if you want to allow the search to continue even when a small percentage of jobs fails.
        Default is 0.1, or 10% or more of jobs failed will raise an exception.
        """
        self.store = store
        self.ml_client = ml_client
        self.timeout = timeout
        self.pipeline_id = pipeline_id
        self.metric_keys = metric_keys
        self.throw_on_failure_rate = throw_on_failure_rate

    def _check_entity_status(self, waiting, completed):
        failed = 0
        for i in range(len(waiting) - 1, -1, -1):
            id = waiting[i]
            e = self.store.get_status(id)
            if self.pipeline_id is None and 'pipeline_id' in e:
                self.pipeline_id = e['pipeline_id']
            if e is not None and 'status' in e and (e['status'] == 'complete' or e['status'] == 'failed'):
                del waiting[i]
                completed[id] = e
                if e['status'] == 'failed':
                    error = e['error']
                    print(f'Training job {id} failed with error: {error}')
                    failed += 1
                else:
                    if len(self.metric_keys) > 0 and self.metric_keys[0] in e:
                        key = self.metric_keys[0]
                        metric = e[key]
                        print(f'Training job {id} completed with {key} = {metric}')
                    else:
                        print(f'Training job {id} completed')
        return failed

    def _get_model_results(self, model_ids, completed):
        # stitch together the models.json file from our status table.
        print('Top model results: ')
        models = []
        interesting_columns = self.metric_keys + ['status', 'error', 'epochs']
        for id in model_ids:
            row = {'id': id}
            e = completed[id] if id in completed else {}
            for key in interesting_columns:
                if key in e:
                    row[key] = e[key]
            models += [row]

        return {
            'models': models
        }

    def _cancel_waiting_list(self, waiting, pipeline_status):
        # cancel any remaining jobs in the waiting list by marking an error status on the entity
        for i in range(len(waiting) - 1, -1, -1):
            id = waiting[i]
            del waiting[i]
            e = self.store.get_status(id)
            if 'error' not in e:
                e['error'] = f'Pipeline {pipeline_status}'
            if 'status' not in e or e['status'] != 'complete':
                e['status'] = pipeline_status.lower()
            self.store.merge_status_entity(e)

    def _get_pipeline_status(self):
        # try and get the status of the Azure ML pipeline, it returns strings like
        # 'Completed', 'Failed', 'Running', 'Preparing', 'Canceled' and so on.
        try:
            if self.pipeline_id is not None:
                train_job = self.ml_client.jobs.get(self.pipeline_id)
                if train_job is not None:
                    return train_job.status
        except Exception as e:
            print(f'Error getting pipeline status for pipeline {self.pipeline_id}: {e}')

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
            failed += self._check_entity_status(waiting, completed)
            if len(waiting) == 0:
                break

            # check the overall pipeline status just in case training jobs failed to even start.
            pipeline_status = self._get_pipeline_status()
            if pipeline_status is not None:
                if pipeline_status == 'Completed':
                    # ok, all jobs are done, which means if we still have waiting tasks then they failed to
                    # even start.
                    self._cancel_waiting_list(waiting, 'failed to start')
                elif pipeline_status == 'Failed' or pipeline_status == 'Canceled':
                    self._cancel_waiting_list(waiting, pipeline_status)

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

        failure_rate = float(failed) / float(len(model_ids))
        if failure_rate > self.throw_on_failure_rate:
            raise Exception(f'Partial Training Pipeline failure rate {failure_rate} exceeds allowed threshold of {self.throw_on_failure_rate}')

        results = self._get_model_results(model_ids, completed)

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
    parser.add_argument('--output', required=True, help='folder to write the results to')
    parser.add_argument('--metrics', type=str, help='metrics to return from the azure table')

    args = parser.parse_args()
    output = args.output
    timeout = args.timeout
    model_path = args.model_path
    metrics = []
    if args.metrics:
        metrics = [x.strip() for x in args.metrics.split(',')]

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

    monitor = JobCompletionMonitor(store, ml_client, metrics, timeout=timeout)
    results = monitor.wait(model_ids)
    if output is not None:
        # save the results with updated validation accuracies to models.json
        with open(os.path.join(output, 'models.json'), 'w') as f:
            f.write(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
