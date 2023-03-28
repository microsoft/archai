# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import logging
import platform
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from azure.storage.blob import BlobServiceClient
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import AsyncModelEvaluator


def _get_utc_date() -> str:
    current_date = datetime.datetime.now()
    current_date = current_date.replace(tzinfo=datetime.timezone.utc)

    return current_date.isoformat()


class RemoteAzureBenchmarkEvaluator(AsyncModelEvaluator):
    """Simple adapter for benchmarking architectures asynchronously on Azure.

    This adapter uploads an ONNX model to a Azure Blob storage container and
    records the model entry on the respective Azure Table.

    """

    def __init__(
        self,
        input_shape: Union[Tuple, List[Tuple]],
        connection_string: str,
        blob_container_name: str,
        table_name: str,
        metric_key: str,
        partition_key: str,
        overwrite: Optional[bool] = True,
        max_retries: Optional[int] = 5,
        retry_interval: Optional[int] = 120,
        onnx_export_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False
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
        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape
        self.sample_input = tuple([torch.rand(*input_shape) for input_shape in input_shapes])
        self.blob_container_name = blob_container_name

        self.table_name = table_name
        self.connection_string = connection_string

        # Changes the Azure logging level to ERROR to avoid unnecessary output
        logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        logger.setLevel(logging.ERROR)

        self.metric_key = metric_key
        self.partition_key = partition_key
        self.overwrite = overwrite
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.onnx_export_kwargs = onnx_export_kwargs or dict()
        self.verbose = verbose

        # Architecture list
        self.archids = []

        # Test connection string
        _ = self.get_table_client()
        _ = self.get_blob_client('test')

    def entry_exists(self, table_client, rowkey_id: str) -> bool:
        try:
            self._get_entity(table_client, rowkey_id)
        except ResourceNotFoundError:
            return False

        return True

    def get_table_client(self) -> Any:
        table_service_client = TableServiceClient.from_connection_string(
            self.connection_string, logging_enable=False, logging_level="ERROR"
        )

        return table_service_client.create_table_if_not_exists(self.table_name)

    def get_blob_client(self, dst_path) -> Any:
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        return blob_service_client.get_blob_client(container=self.blob_container_name, blob=dst_path)

    def _upload_blob(self, src_path: str, dst_path: str) -> None:
        """Upload a file to Azure Blob storage.

        Args:
            src_path: Path to the source file.
            dst_path: Path to the destination file.

        """

        src_path = Path(src_path)
        assert src_path.is_file(), f"{src_path} does not exist or is not a file"

        blob_client = self.get_blob_client(dst_path)

        with open(src_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=self.overwrite)

    def _get_entity(self, table_client, rowkey_id: str) -> Dict[str, Any]:
        """Return the entity with the given `rowkey_id`.

        Args:
            rowkey_id: Row key of the entity.

        Returns:
            Entity dictionary.

        """

        return table_client.get_entity(partition_key=self.partition_key, row_key=rowkey_id)

    def _update_entity(self, table_client, rowkey_id: str, entity_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Update the entity with the given `rowkey_id`.

        Args:
            rowkey_id: Row key of the entity.
            entity_dict: Dictionary containing the entity properties.

        Returns:
            Entity dictionary.

        """

        entity = {"PartitionKey": self.partition_key, "RowKey": rowkey_id}
        entity.update(entity_dict)

        return table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    @overrides
    def send(self, arch: ArchaiModel, dataset_provider: DatasetProvider, budget: Optional[float] = None) -> None:
        archid = str(arch.archid)
        table_client = self.get_table_client()

        # Checks if architecture was already benchmarked
        if self.entry_exists(table_client, archid):
            entity = self._get_entity(table_client, archid)
            
            if self.verbose:
                print(f"Entry for {archid} already exists")

            if entity["status"] == "complete":
                self.archids.append(archid)
                return

            if self.metric_key in entity and entity[self.metric_key]:
                self.archids.append(archid)
                return

        entity = {
            "status": "uploading",
            "name": archid,
            "node": platform.node(),
            "benchmark_only": 1,
            "model_date": _get_utc_date(),
            "model_name": "model.onnx",
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # Uploads ONNX file to blob storage and updates the table entry
            arch.arch.to("cpu")

            # Exports model to ONNX
            torch.onnx.export(
                arch.arch,
                self.sample_input,
                str(tmp_dir / "model.onnx"),
                input_names=[f"input_{i}" for i in range(len(self.sample_input))],
                **self.onnx_export_kwargs,
            )

            self._update_entity(table_client, archid, entity)
            self._upload_blob(str(tmp_dir / "model.onnx"), f"{archid}/model.onnx")

        # Updates model status
        del entity["node"]
        entity["status"] = "new"

        self._update_entity(table_client, archid, entity)
        self.archids.append(archid)

        if self.verbose:
            print(f"Sent {archid} to Remote Benchmark")

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        results = [None for _ in self.archids]

        for _ in range(self.max_retries):
            table_client = self.get_table_client()

            for i, archid in enumerate(self.archids):
                if self.entry_exists(table_client, archid):
                    entity = self._get_entity(table_client, archid)

                    if self.metric_key in entity and entity[self.metric_key]:
                        results[i] = entity[self.metric_key]

            if all(r is not None for r in results):
                break

            if self.verbose:
                status_dict = {
                    "complete": sum(r is not None for r in results),
                    "total": len(results),
                }

                print(
                    f"Waiting for results. Current status: {status_dict}\n"
                    f"Archids: {[archid for archid, status in zip(self.archids, results) if status is None]}"
                )

            time.sleep(self.retry_interval)

        # Resets state
        self.archids = []
        return results
