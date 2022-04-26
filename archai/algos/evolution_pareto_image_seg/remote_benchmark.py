import os
from pathlib import Path
from typing import Dict, List, Optional

from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, UpdateMode


class RemoteAzureBenchmark():
    def __init__(self, connection_string: str,
                 blob_container_name: str,
                 table_name: str,
                 partition_key: str,
                 metrics: List[str],
                 overwrite: bool = True):
        """ Simple adapter for sending and reading from Azure Blob Storage and Table Storage.

        Args:
            connection_string (str): Storage account connection string
            blob_container_name (str): Name of the blob container
            table_name (str): Name of the table
            partition_key (str): Partition key for the table
            metrics (List[str]): List of metrics to be read from the table
            overwrite (bool, optional): Whether to overwrite existing models. Defaults to True.
        """

        self.blob_container_name = blob_container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        self.table_name = table_name
        self.table_service_client = TableServiceClient.from_connection_string(connection_string)
        self.partition_key = partition_key

        self.metrics = metrics
        self.overwrite = overwrite

    @property
    def table_client(self):
        return self.table_service_client.create_table_if_not_exists(
            self.table_name
        ) 

    def upload_model(self, model_id: str, onnx_path: str) -> None:
        onnx_path = Path(onnx_path)
        assert onnx_path.exists(), f'{onnx_path} does not exist'

        blob_client = self.blob_service_client.get_blob_client(
            container=self.blob_container_name,
            blob=f'{model_id}/{str(onnx_path)}'
        )

        with open(onnx_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=self.overwrite)

    def update_entity(self, model_id: str, entity_dict: Dict) -> Dict:
        entity = {
            'PartitionKey': self.partition_key,
            'RowKey': model_id
        }

        entity.update(entity_dict)
        return self.table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    def get_metrics(self, model_id: str) -> Dict:
        entity = self.table_client.get_entity(
            partition_key=self.partition_key, row_key=model_id
        )

        return {
            k: v for k, v in entity.items() 
            if k in ['PartitionKey', 'RowKey'] + self.metrics
        }
