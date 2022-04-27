import logging
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, UpdateMode

from archai.nas.arch_meta import ArchWithMetaData
from archai.algos.evolution_pareto_image_seg.utils import to_onnx, get_utc_date

class RemoteAzureBenchmark():
    def __init__(self, connection_string: str,
                 blob_container_name: str,
                 table_name: str,
                 partition_key: str,
                 overwrite: bool = True):
        """ 
            Simple adapter for benchmarking architectures asynchronously on Azure.
            This adapter uploads an Architecture to a Azure Blob storage container and 
            records the model entry on the respective Azure Table. 

        Args:
            connection_string (str): Storage account connection string
            blob_container_name (str): Name of the blob container
            table_name (str): Name of the table
            partition_key (str): Partition key for the table used to record all entries
            overwrite (bool, optional): Whether to overwrite existing models. Defaults to True.
        """

        self.blob_container_name = blob_container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        self.table_name = table_name
        self.table_service_client = TableServiceClient.from_connection_string(
            connection_string, logging_enable=False, logging_level='ERROR'
        )
        
        # Changes the Azure logging level to ERROR to avoid unnecessary output
        logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
        logger.setLevel(logging.ERROR)

        self.partition_key = partition_key
        self.overwrite = overwrite

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

    def send_model(self, model: ArchWithMetaData) -> None:
        archid = str(model.metadata['archid'])
        entity = {
            'status': 'uploading', 'name': archid, 'node': platform.node(),
            'benchmark_only': 1, 'model_date': get_utc_date()
        }
        
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # Uploads ONNX file to blob storage and updates the table entry
            to_onnx(
                model=model.arch, img_size=model.arch.img_size,
                output_path=tmp_dir / 'model.onnx',
            )
            
            self.update_entity(archid, entity)
            self.upload_blob(tmp_dir / 'model.onnx', f'{archid}/model.onnx')

            # Uploads the model architecture as well
            model.arch.to_file(tmp_dir / 'architecture.yaml')
            self.upload_blob(tmp_dir / 'architecture.yaml', f'{archid}/architecture.yaml')

        # Updates model status
        del entity['node']
        entity['status'] = 'new'

        self.update_entity(archid, entity)
