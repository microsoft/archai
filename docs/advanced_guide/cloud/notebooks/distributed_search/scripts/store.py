# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import re
import logging
import datetime
import platform
from azure.data.tables import TableServiceClient, UpdateMode, EntityProperty, EdmType
from azure.storage.blob import BlobClient, ContainerClient

"""ArchaiStore wraps the Azure 'status' Table and associated Blob Storage used to provide a backing
store and collated status for long running Archai search jobs."""
class ArchaiStore:
    def __init__(self, storage_account_key, storage_account_name, blob_container_name='models', status_table_name='status'):
        self.storage_account_key = storage_account_key
        self.storage_account_name = storage_account_name
        self.storage_connection_string = f'DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net'
        self.blob_container_name = blob_container_name
        self.status_table_name = status_table_name
        self.service = None
        self.table_client = None
        self.container_client = None

    @staticmethod
    def parse_connection_string(storage_connection_string):
        parts = storage_connection_string.split(";")
        storage_account_name = None
        storage_account_key = None
        for part in parts:
            keyvalue = part.split("=")
            key = keyvalue[0]
            if key == "AccountName":
                storage_account_name = keyvalue[1]
            elif key == "AccountKey":
                storage_account_key = keyvalue[1]

        if not storage_account_name:
            raise Exception("storage_connection_string is missing AccountName part")

        if not storage_account_key:
            raise Exception("storage_connection_string is missing AccountKey part")

        return (storage_account_name, storage_account_key)

    def get_utc_date(self):
        current_date = datetime.datetime.now()
        current_date = current_date.replace(tzinfo=datetime.timezone.utc)
        return current_date.isoformat()

    def _get_node_id(self):
        return platform.node()

    def _get_status_table_service(self):
        logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
        logger.setLevel(logging.ERROR)
        return TableServiceClient.from_connection_string(conn_str=self.storage_connection_string, logger=logger, logging_enable=False)

    def _get_table_client(self):
        if not self.table_client:
            if not self.service:
                self.service = self._get_status_table_service()
            self.table_client = self.service.create_table_if_not_exists(self.status_table_name)
        return self.table_client

    def _get_container_client(self, name):
        if not self.container_client:
            logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
            logger.setLevel(logging.ERROR)
            self.container_client = ContainerClient.from_connection_string(
                self.storage_connection_string,
                container_name=name,
                logger=logger,
                logging_enable=False)
            if not self.container_client.exists():
                self.container_client.create_container()
        return self.container_client

    def _get_blob_client(self, name):
        container = self._get_container_client(self.blob_container_name)  # make sure container exists.
        return BlobClient.from_connection_string(self.storage_connection_string, container_name=container.container_name, blob_name=name)

    def get_all_status_entities(self, status=None, not_equal=False):
        """ Get all status entities with optional status column filter.
        For example, pass "status=complete" to find all status rows that
        have the status "complete".  Pass not_equal of True if you want
        to check the status is not equal to the given value.
        """
        table_client = self._get_table_client()

        entities = []
        query = "PartitionKey eq 'main'"
        if status:
            if not_equal:
                query += f" and status ne '{status}'"
            else:
                query += f" and status eq '{status}'"

        try:
            # find all properties (just use list_entities?)
            for e in table_client.query_entities(query_filter=query):
                entities += [e]

        except Exception as e:
            print(f"### error reading table: {e}")

        return entities

    def get_status(self, name):
        table_client = self._get_table_client()

        try:
            entity = table_client.get_entity(partition_key='main', row_key=name)
        except Exception:
            entity = {
                'PartitionKey': 'main',
                'RowKey': name,
                'name': name,
                'status': 'new'
            }
        return entity

    def get_existing_status(self, name):
        table_client = self._get_table_client()
        try:
            return table_client.get_entity(partition_key='main', row_key=name)
        except Exception:
            return None

    def update_status_entity(self, entity):
        table_client = self._get_table_client()
        table_client.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)

    def merge_status_entity(self, entity):
        table_client = self._get_table_client()
        table_client.update_entity(entity=entity, mode=UpdateMode.MERGE)

    def update_status(self, name, status, priority=None):
        table_client = self._get_table_client()

        try:
            entity = table_client.get_entity(partition_key='main', row_key=name)
        except Exception:
            entity = {
                'PartitionKey': 'main',
                'RowKey': name,
                'name': name,
                'status': status
            }

        entity['status'] = status
        if priority:
            entity['priority'] = priority
        self.update_status_entity(entity)
        return entity

    def delete_status(self, name):
        table_client = self._get_table_client()

        for e in self.get_all_status_entities():
            if 'name' in e and e['name'] == name:
                print(f"Deleting status entity for {name}")
                table_client.delete_entity(e)

    def upload_blob(self, name, file, blob_name=None):
        filename = os.path.basename(file)
        if blob_name:
            blob = f"{name}/{blob_name}"
        else:
            blob = f"{name}/{filename}"

        blob_client = self._get_blob_client(blob)

        with open(file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    def lock(self, name, status):
        e = self.get_status(name)
        node_id = self._get_node_id()
        if 'node' in e and e['node'] != node_id:
            print(f"This model is locked by {e['node']}")
            return
        e['status'] = status
        e['node'] = self._get_node_id()  # lock the row until upload complete
        self.update_status_entity(e)
        return e

    def unlock(self, name):
        e = self.get_status(name)
        if 'node' in e:
            del e['node']
            self.update_status_entity(e)
        return e

    def unlock_all(self, node_name):
        # fix up the 'date' property for uploaded blobs
        for e in self.get_all_status_entities():
            name = e['name']
            node = e['node'] if 'node' in e else None
            changed = False
            if 'node' in e:
                if node_name and node_name != node:
                    continue
                e['node'] = ''
                changed = True

            if changed:
                print(f"Unlocking job {name} on node {node}")
                self.merge_status_entity(e)

    def upload(self, name, path, **kwargs):
        if not name:
            raise Exception('Model name is missing')
        e = self.get_status(name)

        e = self.lock(name, 'uploading')
        self.update_status_entity(e)

        try:
            to_upload = []
            if os.path.isdir(path):
                to_upload = [os.path.join(path, f) for f in os.listdir(path)]
            elif os.path.isfile(path):
                to_upload = [path]
            else:
                raise Exception(f'Path not found: {path}')

            for f in to_upload:
                # upload the file
                print(f'Uploading file: {f} to blob: {name}')
                self.upload_blob(name, f)
        except Exception as ex:
            print(f'### upload failed: {ex}')

        # record uploaded status and unlock it.
        self.reset_metrics(e)
        e['status'] = 'uploaded'
        del e['node']
        for k in kwargs:
            e[k] = kwargs[k]

        self.update_status_entity(e)

    def download(self, friendly_name, folder, specific_file=None, all_files=False, no_dlc=False):
        """ Download files from the given friendly name folder of the blob_container_name.
        and return the local path to that file including the folder.  If an optional specific_file is
        given then it tries to find and download that file only.  If you set all_files to true
        it will download all files associated with the friendly name.  If friendly_name is empty
        it will download everything from the blob store. """
        container = self._get_container_client(self.blob_container_name)
        if not container.exists():
            return (False, None)

        if not os.path.isdir(folder):
            os.makedirs(folder)
        model_found = False
        model_name = None
        local_file = None
        prefix = f'{friendly_name}/'
        supported = ['.onnx', '.dlc', '.pt', '.pd']
        if no_dlc:
            supported = ['.onnx', '.pt', '.pd']

        for blob in container.list_blobs(name_starts_with=prefix):
            model_name = blob.name[len(prefix):]
            parts = os.path.splitext(model_name)
            download = False
            if all_files:
                download = True
                local_file = os.path.join(folder, model_name)
            elif specific_file:
                if specific_file != model_name:
                    continue
                else:
                    download = True
                    local_file = os.path.join(folder, model_name)

            elif len(parts) > 1:
                filename, ext = parts
                if ext in supported:
                    if '.quant' in filename:
                        local_file = os.path.join(folder, 'model.quant' + ext)
                    else:
                        local_file = os.path.join(folder, 'model' + ext)
                    download = True

            if download:
                print(f"Downloading file: {model_name} to {local_file} ...")
                blob_client = container.get_blob_client(blob)
                with open(local_file, 'wb') as f:
                    data = blob_client.download_blob()
                    f.write(data.readall())
                model_found = True
                if not all_files:
                    break

        return (model_found, model_name, local_file)

    def delete_blobs(self, name, specific_file=None):
        container = self._get_container_client(self.blob_container_name)
        prefix = f'{name}/'
        for blob in container.list_blobs(name_starts_with=prefix):
            file_name = blob.name[len(prefix):]
            if specific_file and file_name != specific_file:
                continue
            container.delete_blob(blob)

    def reset_metrics(self, entity):
        # now clear all data to force a full re-run of everything.
        for key in ['accuracy', 'precision', 'recall', 'f1', 'mean', 'stdev', 'total_inference_avg', 'macs', 'parameters']:
            if key in entity:
                del entity[key]

    def print_entities(self, entities, columns=None):
        keys = []
        for e in entities:
            for k in e:
                if k not in keys and k != 'PartitionKey' and k != 'RowKey':
                    if columns is None or k in columns:
                        keys += [k]

        # so we can convert to .csv format
        print(", ".join(keys))
        for e in entities:
            for k in keys:
                if k in e:
                    x = e[k]
                    if isinstance(x, EntityProperty) and x.edm_type is EdmType.INT64:
                        x = x.value
                    v = str(x).replace(',', ' ').replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                    print(f"{v}", end='')
                print(', ', end='')
            print()


def status(con_str, args):
    parser = argparse.ArgumentParser(description='Print status in .csv format')
    parser.add_argument('--status', help='Optional match for the status column (default None).')
    parser.add_argument('--name', help='Optional name of single status row to return (default None).')
    parser.add_argument('--not_equal', '-ne', help='Switch the match to a not-equal comparison.', action="store_true")
    parser.add_argument('--locked', help='Find entities that are locked by a node.', action="store_true")
    parser.add_argument('--cols', help='Comma separated list of columns to report (default is to print all)')
    args = parser.parse_args(args)
    store = ArchaiStore(con_str)
    entities = store.get_all_status_entities(args.status, args.not_equal)
    if args.locked:
        entities = [e for e in entities if 'node' in e and e['node']]
    if args.name:
        entities = [e for e in entities if 'name' in e and e['name'] == args.name]

    columns = None
    if args.cols:
        columns = [x.strip() for x in args.cols.split(',')]
    store.print_entities(entities, columns)


def upload(con_str, args):
    parser = argparse.ArgumentParser( description='Upload a model to azure blob store')
    parser.add_argument('name', help='Friendly name of the folder to put this in.')
    parser.add_argument('file', help='Path to the file to upload to Azure or a folder to upload all files in that folder.')
    parser.add_argument('--priority', type=int, help='Optional priority override for this job. ' +
                             'Larger numbers mean lower priority')
    args = parser.parse_args(args)
    store = ArchaiStore(con_str)
    store.upload(args.name, args.file, priority=args.priority)


def download(con_str, args):
    parser = argparse.ArgumentParser(
        description="Download assets from azure blob store using friendly name.")
    parser.add_argument('--name', help='Friendly name of model to download (if not provided it downloads them all')
    parser.add_argument('--file', help='The optional name of the files to download instead of getting them all.')
    args = parser.parse_args(args)

    store = ArchaiStore(con_str)
    friendly_name = args.name
    if not friendly_name:
        friendly_names = [e['name'] for e in store.get_all_status_entities()]
    else:
        friendly_names = [friendly_name]

    specific_file = args.file
    all_files = False if specific_file else True

    for friendly_name in friendly_names:
        found, model, file = store.download(friendly_name, friendly_name, specific_file, all_files)
        if not found and specific_file:
            print(f"file {specific_file} not found")


def delete(con_str, args):
    parser = argparse.ArgumentParser(description='Delete a model from azure using its friendly name')
    parser.add_argument('name', help='The friendly name allocated by the upload script.')
    parser.add_argument('--file', help='Delete just the one file associated with the friendly name.')
    args = parser.parse_args(args)

    store = ArchaiStore(con_str)
    store.delete_blobs(args.name, args.file)
    if not args.file:
        store.delete_status(args.name)


def unlock(con_str, args):
    parser = argparse.ArgumentParser(
        description='Unlock all jobs for given node or unlock all jobs.')
    parser.add_argument('--node', help='Optional node name (default None).')
    args = parser.parse_args(args)
    store = ArchaiStore(con_str)
    store.unlock_all(args.node)

if __name__ == '__main__':
    CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    if len(sys.argv) <= 1:
        print("Expecting a command, one of 'status', 'upload', 'delete', 'download', 'unlock'")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    store = ArchaiStore(con_str)
    if cmd == 'status':
        status(con_str, args)
    elif cmd == 'upload':
        upload(con_str, args)
    elif cmd == 'download':
        download(con_str, args)
    elif cmd == 'delete':
        delete(con_str, args)
    elif cmd == 'unlock':
        unlock(con_str, args)
    else:
        print(f"Unknown command: {cmd}, expecting one of status, upload, download, delete, lock, unlock")
        sys.exit(1)