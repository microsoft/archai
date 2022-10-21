import logging
import platform
import time
import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union, Tuple, Optional, Any
from overrides import overrides

import torch
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, UpdateMode

from archai.discrete_search import AsyncObjective, DatasetProvider, ArchaiModel


def get_utc_date():
    current_date = datetime.datetime.now()
    current_date = current_date.replace(tzinfo=datetime.timezone.utc)
    return current_date.isoformat()


class RemoteAzureBenchmarkObjective(AsyncObjective):
    def __init__(self, 
                 input_shape: Union[Tuple, List[Tuple]],
                 connection_string: str,
                 blob_container_name: str,
                 table_name: str,
                 metric_key: str,
                 partition_key: str,
                 overwrite: bool = True,
                 max_retries: int = 5,
                 retry_interval: int = 120,
                 onnx_export_kwargs: Optional[Dict[str, Any]] = None):
        """ 
            Simple adapter for benchmarking architectures asynchronously on Azure.
            This adapter uploads an ONNX model to a Azure Blob storage container and 
            records the model entry on the respective Azure Table. 

        Args:
            input_shape (Union[Tuple, List[Tuple]]): Model Input shape or list of model input shapes for ONNX export.
            connection_string (str): Storage account connection string
            blob_container_name (str): Name of the blob container
            table_name (str): Name of the table
            metric_key (str): Column that should be used as result
            partition_key (str): Partition key for the table used to record all entries
            overwrite (bool, optional): Whether to overwrite existing models. Defaults to True.
            max_retries (int, optional): Maximum number of retries in `fetch_all`.
            retry_interval (int, optional): Interval between each retry attempt.
            onnx_export_kwargs (Dict, optional): Dictionary containing key-value arguments for `torch.onnx.export`
        """
        # TODO: Make this class more general / less pipeline-specific

        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape        
        self.sample_input = tuple([torch.rand(*input_shape) for input_shape in input_shapes])
        self.blob_container_name = blob_container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        self.table_name = table_name
        self.table_service_client = TableServiceClient.from_connection_string(
            connection_string, logging_enable=False, logging_level='ERROR'
        )
        
        # Changes the Azure logging level to ERROR to avoid unnecessary output
        logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
        logger.setLevel(logging.ERROR)

        self.metric_key = metric_key
        self.partition_key = partition_key
        self.overwrite = overwrite
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.onnx_export_kwargs = onnx_export_kwargs or dict()

        # Architecture list
        self.archids = []

    @property
    def table_client(self):
        return self.table_service_client.create_table_if_not_exists(
            self.table_name
        ) 

    def __contains__(self, rowkey_id: str):
        try:
            self.get_entity(rowkey_id)
        except ResourceNotFoundError:
            return False
        
        return True

    def upload_blob(self, src_path: str, dst_path: str) -> None:
        src_path = Path(src_path)
        assert src_path.is_file(), f'{src_path} does not exist or is not a file'

        blob_client = self.blob_service_client.get_blob_client(
            container=self.blob_container_name, blob=dst_path
        )

        with open(src_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=self.overwrite)

    def get_entity(self, rowkey_id: str) -> Dict:
        return self.table_client.get_entity(
            partition_key=self.partition_key, row_key=rowkey_id
        )

    def update_entity(self, rowkey_id: str, entity_dict: Dict) -> Dict:
        entity = {
            'PartitionKey': self.partition_key,
            'RowKey': rowkey_id
        }

        entity.update(entity_dict)
        return self.table_client.upsert_entity(entity, mode=UpdateMode.REPLACE)

    @overrides
    def send(self, nas_model: ArchaiModel, dataset_provider: DatasetProvider,
             budget: Optional[float] = None) -> None:
        archid = str(nas_model.archid)

        # Checks if architecture was already benchmarked
        if archid in self:
            entity = self.get_entity(archid)
            if entity['status'] == 'complete':
                self.archids.append(archid)
                return
            
            if self.metric_key in entity and entity[self.metric_key]:
                self.archids.append(archid)
                return

        entity = {
            'status': 'uploading', 'name': archid, 'node': platform.node(),
            'benchmark_only': 1, 'model_date': get_utc_date(),
            'model_name': 'model.onnx'
        }
        
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # Uploads ONNX file to blob storage and updates the table entry
            nas_model.arch.to('cpu')

            # Exports model to ONNX
            torch.onnx.export(
                nas_model.arch, self.sample_input, str(tmp_dir / 'model.onnx'),
                input_names=[f'input_{i}' for i in range(len(self.sample_input))],
                **self.onnx_export_kwargs
            )
            
            self.update_entity(archid, entity)
            self.upload_blob(str(tmp_dir / 'model.onnx'), f'{archid}/model.onnx')

        # Updates model status
        del entity['node']
        entity['status'] = 'new'

        self.update_entity(archid, entity)
        self.archids.append(archid)

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        results = [None for _ in self.archids]
        
        for _ in range(self.max_retries):
            for i, archid in enumerate(self.archids):
                if archid in self:
                    entity = self.get_entity(archid)
                    
                    if self.metric_key in entity and entity[self.metric_key]:
                        results[i] = entity[self.metric_key]
            
            if all(r is not None for r in results):
                break

            time.sleep(self.retry_interval)

        # Resets state
        self.archids = []
        return results
