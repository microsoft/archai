# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import datetime
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from overrides import overrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator
from archai.common.store import ArchaiStore


class RemoteAzureBenchmarkEvaluator(AsyncModelEvaluator):
    """Simple adapter for benchmarking architectures asynchronously on Azure.

    This adapter uploads an ONNX model to a Azure Blob storage container and
    records the model entry on the respective Azure Table.

    """

    def __init__(
        self,
        input_shape: Union[Tuple, List[Tuple]],
        store: ArchaiStore,
        experiment_name: str,
        metric_key: str,
        overwrite: Optional[bool] = True,
        max_retries: Optional[int] = 5,
        retry_interval: Optional[int] = 120,
        onnx_export_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        benchmark_only: bool = True
    ) -> None:
        """Initialize the evaluator.

        Args:
            input_shape: Model input shape or list of model input shapes for ONNX export.
            connection_string: Storage account connection string.
            blob_container_name: Name of the blob container.
            table_name: Name of the table.
            metric_key: Column that should be used as result.
            partition_key: Partition key for the table used to record all entries.
            overwrite: Whether to overwrite existing models.
            max_retries: Maximum number of retries in `fetch_all`.
            retry_interval: Interval between each retry attempt.
            onnx_export_kwargs: Dictionary containing key-value arguments for `torch.onnx.export`.
            verbose: Whether to print debug messages.
        """

        # TODO: Make this class more general / less pipeline-specific
        self.store = store
        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape
        self.sample_input = tuple([torch.rand(*input_shape) for input_shape in input_shapes])
        self.experiment_name = experiment_name
        self.metric_key = metric_key
        self.overwrite = overwrite
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.onnx_export_kwargs = onnx_export_kwargs or dict()
        self.verbose = verbose
        self.results = {}
        self.benchmark_only = benchmark_only

        # Architecture list
        self.archids = []

        # Test connection string works
        unknown_id = str(uuid.uuid4())
        _ = self.store.get_existing_status(unknown_id)
        _ = self.store.list_blobs(unknown_id)

    def _reset(self, entity):
        changed = False
        for k in ['mean', 'macs', 'params', 'stdev', 'total_inference_avg', 'error']:
            if k in entity:
                del entity[k]
                changed = True
        if changed:
            self.store.update_status_entity(entity)

    @overrides
    def send(self, arch: ArchaiModel, budget: Optional[float] = None) -> None:
        # bug in azure ml sdk requires blob store folder names not begin with digits, so we prefix with 'id_'
        archid = f'id_{arch.archid}'

        # Checks if architecture was already benchmarked
        entity = self.store.get_existing_status(archid)
        if entity is not None:
            if entity["status"] == "complete":
                if self.metric_key in entity:
                    if self.verbose:
                        value = entity[self.metric_key]
                        self.archids.append(archid)
                        print(f"Entry for {archid} already exists with {self.metric_key} = {value}")
                    return
                else:
                    # complete but missing the mean, so reset the benchmark metrics so we can try again.
                    self._reset(entity)
            else:
                # job is still running, let it continue
                if self.verbose:
                    print(f"Job for {archid} is running...")
                self.archids.append(archid)
                return

        entity = self.store.get_status(archid)  # this is a get or create operation.
        if self.benchmark_only:
            entity["benchmark_only"] = 1
        entity["model_date"] = self.store.get_utc_date()
        entity["model_name"] = "model.onnx"
        self.store.update_status_entity(entity)  # must be an update, not a merge.
        self.store.lock_entity(entity, "uploading")
        try:
            with TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)

                # Uploads ONNX file to blob storage and updates the table entry
                arch.arch.to("cpu")
                file_name = str(tmp_dir / "model.onnx")
                # Exports model to ONNX
                torch.onnx.export(
                    arch.arch,
                    self.sample_input,
                    file_name,
                    input_names=[f"input_{i}" for i in range(len(self.sample_input))],
                    **self.onnx_export_kwargs,
                )

                self.store.upload_blob(f'{self.experiment_name}/{archid}', file_name, "model.onnx")
                entity["status"] = "new"
        except Exception as e:
            entity["error"] = str(e)
        finally:
            self.store.unlock_entity(entity)

        self.archids.append(archid)

        if self.verbose:
            print(f"Sent {archid} to Remote Benchmark")

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        results = [None] * len(self.archids)
        completed = [False] * len(self.archids)

        # retries defines how long we wait for progress, as soon as we see something complete we
        # reset this counter because we are making progress.
        retries = self.max_retries
        start = time.time()
        count = 0
        while retries > 0:
            for i, archid in enumerate(self.archids):
                if not completed[i]:
                    entity = self.store.get_existing_status(archid)
                    if entity is not None:
                        if self.metric_key in entity and entity[self.metric_key]:
                            results[i] = entity[self.metric_key]
                        if "error" in entity:
                            error = entity["error"]
                            print(f"Skipping architecture {archid} because of remote error: {error}")
                            completed[i] = True
                            retries = self.max_retries
                        elif entity["status"] == "complete":
                            print(f"Architecture {archid} is complete with {self.metric_key}={ results[i]}")
                            completed[i] = True
                            retries = self.max_retries

            if all(completed):
                break

            count = sum(1 for c in completed if c)

            if self.verbose:
                remaining = len(self.archids) - count
                estimate = (time.time() - start) / (count + 1) * remaining

                status_dict = {
                    "complete": count,
                    "total": len(results),
                    "time_remaining": str(datetime.timedelta(seconds=estimate))
                }

                print(
                    f"Waiting for results. Current status: {status_dict}\n"
                    f"Pending Archids: {[archid for archid, status in zip(self.archids, results) if status is None]}"
                )

            time.sleep(self.retry_interval)
            retries += 1

        count = sum(1 for c in completed if c)
        if count == 0:
            raise Exception("Something is wrong, the uploaded models are not being processed. Please check your SNPE remote runner setup.")

        # Resets state
        self.archids = []
        return results
