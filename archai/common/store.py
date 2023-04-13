# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import glob
import sys
import logging
import datetime
import platform
import numpy as np
import re
from torch import Tensor
from azure.data.tables import TableServiceClient, UpdateMode, EntityProperty, EdmType
from azure.storage.blob import BlobClient, ContainerClient
from shutil import rmtree


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


class ArchaiStore:
    """ArchaiStore wraps an Azure 'status' Table and associated Blob Storage used to provide a backing
    store for models and an associated table for collating status of long running jobs.  This is actually a general
    purpose utility class that could be used for anything.

    The naming scheme is such that each Entity in the table has a 'name' property which is a simple friendly name or a
    guid, and this row will have an associated folder in the blob storage container with the same name where models and
    other peripheral files can be stored.

    The 'status' table supports a locking concept that allows the status table to be used as a way of coordinating jobs
    across multiple machines where each machine grabs free work, locks that row until the work is done, uploads new
    files, and updates the status to 'complete' then unlocks that row. So this ArchaiStore can be used as the backing
    store for a simple distributed job scheduler.

    This also has a convenient command line interface provided below.
    """
    def __init__(self, storage_account_name, storage_account_key, blob_container_name='models', table_name='status', partition_key='main'):
        self.storage_account_key = storage_account_key
        self.storage_account_name = storage_account_name
        self.storage_connection_string = f'DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net'
        self.blob_container_name = blob_container_name
        self.status_table_name = table_name
        self.partition_key = partition_key
        self.service = None
        self.table_client = None
        self.container_client = None

    @staticmethod
    def parse_connection_string(storage_connection_string):
        """ This helper method extracts the storage account name and key pair from a connection string
        and returns that pair in a tuple.  This pair can then be used to construct an ArchaiStore object """
        parts = storage_connection_string.split(";")
        storage_account_name = None
        storage_account_key = None
        for part in parts:
            i = part.find('=')
            key = part[0:i]
            value = part[i + 1:]
            if key == "AccountName":
                storage_account_name = value
            elif key == "AccountKey":
                storage_account_key = value

        if not storage_account_name:
            raise Exception("storage_connection_string is missing AccountName part")

        if not storage_account_key:
            raise Exception("storage_connection_string is missing AccountKey part")

        return (storage_account_name, storage_account_key)

    def get_utc_date(self):
        """ This handy function can be used to put a UTC timestamp column in your entity, like a 'model_date' column, for example. """
        current_date = datetime.datetime.now()
        current_date = current_date.replace(tzinfo=datetime.timezone.utc)
        return current_date.isoformat()

    def _get_node_id(self):
        """ Return a unique name for the current machine which is used as the lock identity """
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
        query = f"PartitionKey eq '{self.partition_key}'"
        if status:
            if not_equal:
                query += f" and status ne '{status}'"
            else:
                query += f" and status eq '{status}'"

        try:
            # find all properties (just use list_entities?)
            for e in table_client.query_entities(query_filter=query):
                entities += [self._unwrap_numeric_types(e)]

        except Exception as e:
            print(f"### error reading table: {e}")

        return entities

    def get_status(self, name):
        """ Get or create a new status entity with the given name.
        The returned entity is a python dictionary where the name can be retrieved
        using e['name'], you can then add keys to that dictionary and call update_status_entity. """
        table_client = self._get_table_client()

        try:
            entity = table_client.get_entity(partition_key=self.partition_key, row_key=name)
            entity = self._unwrap_numeric_types(entity)
        except Exception:
            entity = {
                'PartitionKey': self.partition_key,
                'RowKey': name,
                'name': name,
                'status': 'new'
            }
            self.update_status_entity(entity)
        return entity

    def _wrap_numeric_types(self, entity):
        e = {}
        for k in entity.keys():
            v = entity[k]
            if isinstance(v, bool):
                e[k] = v
            elif isinstance(v, int):
                e[k] = EntityProperty(v, EdmType.INT64)
            elif isinstance(v, float):
                e[k] = float(v)  # this is casting np.float to float.
            else:
                e[k] = v
        return e

    def _unwrap_numeric_types(self, entity):
        e = {}
        for k in entity.keys():
            v = entity[k]
            if isinstance(v, EntityProperty):
                e[k] = v.value
            else:
                e[k] = v
        return e

    def get_existing_status(self, name):
        """ Find the given entity by name, and return it, or return None if the name is not found."""
        table_client = self._get_table_client()
        try:
            entity = table_client.get_entity(partition_key=self.partition_key, row_key=name)
            entity = self._unwrap_numeric_types(entity)
        except Exception:
            return None
        return entity

    def get_updated_status(self, e):
        """ Return an updated version of the entity by querying the table again, this way you
        can pick up any changes that another process may have made. """
        table_client = self._get_table_client()
        try:
            entity = table_client.get_entity(partition_key=self.partition_key, row_key=e['RowKey'])
            entity = self._unwrap_numeric_types(entity)
        except Exception:
            return None
        return entity

    def update_status_entity(self, entity):
        """ This method replaces everything in the entity store with what you have here.
        The entity can store strings, bool, float, int, datetime, so anything like a python list
        is best serialized using json.dumps and stored as a string, the you can use json.loads to
        parse it later. """
        # Note that for things larger than Int32 we need to use EntityProperty with EdmType.INT64, and
        # so we do that automatically here for the user so they don't have to, and get entity will turn
        # the result back into a python integer than can be larger than Int32.
        table_client = self._get_table_client()
        entity = self._wrap_numeric_types(entity)
        table_client.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)

    def merge_status_entity(self, entity):
        """ This method merges everything in the entity store with what you have here. So you can
        add a property without clobbering any other new properties other processes have added in
        parallel.  Merge cannot delete properties, for that you have to use update_status_entity.

        The entity can store strings, bool, float, int, datetime, so anything like a python list
        is best serialized using json.dumps and stored as a string, the you can use json.loads to
        parse it later."""
        table_client = self._get_table_client()
        entity = self._wrap_numeric_types(entity)
        table_client.update_entity(entity=entity, mode=UpdateMode.MERGE)

    def update_status(self, name, status, priority=None):
        """ This is a simple wrapper that gets the entity by name, and updates the status field.
        If you already have the entity then use update_status_entity."""
        entity = self.get_existing_status(name)
        if entity is None:
            entity = self.get_status(name)
            self.update_status_entity(entity)

        entity['status'] = status
        if priority:
            entity['priority'] = priority
        self.merge_status_entity(entity)
        return entity

    def delete_status(self, name):
        """ Delete the status entry with this name, note this does not delete any associated blobs.
        See delete_blobs for that.  """
        e = self.get_existing_status(name)
        if e is not None:
            table_client = self._get_table_client()
            table_client.delete_entity(e)

    def delete_status_entity(self, e):
        """ Delete the status entry with this name, note this does not delete any associated blobs.
        See delete_blobs for that.  """
        table_client = self._get_table_client()
        e = self.get_existing_status(e['name'])
        if e is not None:
            table_client.delete_entity(e)

    def upload_blob(self, folder_name, file, blob_name=None):
        """ Upload the given file to the blob store, under the given folder name.
        The folder name could have multiple parts like 'project/experiment/foo'.
        By default the blob will use the base file name, but you can override
        that with the given blob_name if you want to.  """
        filename = os.path.basename(file)
        if blob_name:
            blob = f"{folder_name}/{blob_name}"
        else:
            blob = f"{folder_name}/{filename}"

        blob_client = self._get_blob_client(blob)

        with open(file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    def lock(self, name, status):
        """ Lock the named entity to this computer identified by platform.node()
        and set the status to the given status.  This way you can use this ArchaiStore as
        a way of coordinating the parallel executing of a number of jobs, where each long
        running job is allocated to a particular node in a distributed cluster using this
        locking mechanism.  Be sure to call unlock when done, preferably in a try/finally block. """
        e = self.get_existing_status(name)
        if e is None:
            e = self.get_status(name)
            self.update_status_entity(e)
        return self.lock_entity(e, status)

    def lock_entity(self, e, status):
        """ Lock the given entity to this computer identified by platform.node()
        and set the status to the given status.  This way you can use this ArchaiStore as
        a way of coordinating the parallel executing of a number of jobs, where each long
        running job is allocated to a particular node in a distributed cluster using this
        locking mechanism.  Be sure to call unlock when done, preferably in a try/finally block. """
        node_id = self._get_node_id()
        if 'node' in e and e['node'] != node_id:
            name = e['name']
            print(f"The model {name} is locked by {e['node']}")
            return None
        e['status'] = status
        e['node'] = node_id  # lock the row until upload complete
        self.merge_status_entity(e)
        return e

    def is_locked(self, name):
        """ Return true if the entity exists and is locked by anyone (including this computer). """
        e = self.get_existing_status(name)
        if e is None:
            return False
        return 'node' in e

    def is_locked_by_self(self, name):
        """ Return true if the entity exists and is locked this computer.  This is handy if the
        computer restarts and wants to continue processing rows it has already claimed. """
        e = self.get_existing_status(name)
        if e is None:
            return False
        node_id = self._get_node_id()
        return 'node' in e and e['node'] == node_id

    def is_locked_by_other(self, name):
        """ Return true if the entity exists and is locked some other computer.  This will tell
        the local computer not to touch this row of the table as someone else has it. """
        e = self.get_existing_status(name)
        if e is None:
            return False
        node_id = self._get_node_id()
        return 'node' in e and e['node'] != node_id

    def unlock(self, name):
        """ Unlock the entity (regardless of who owns it - so use carefully, preferably only
        when is_locked_by_self is true). """
        e = self.get_status(name)
        self.unlock_entity(e)
        return e

    def unlock_entity(self, e):
        """ Unlock the entity (regardless of who owns it - so use carefully, preferably only
        when is_locked_by_self is true). """
        if 'node' in e:
            del e['node']
            self.update_status_entity(e)
        else:
            self.merge_status_entity(e)
        return e

    def get_lock(self, entity):
        """ Find out what computer has the entity locked. """
        if 'node' in entity and entity['node']:
            return entity['node']
        return None

    def unlock_all(self, node_name):
        """ This is a sledge hammer for unlocking all entities, use carefully.
        This might be necessary if you are moving everything to a new cluster. """
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

    def reset(self, name, except_list=[]):
        """ This resets all properties on the given entity that are not primary keys,
        'name' or 'status' and are not in the given except_list.
        This will not touch a node that is locked by another computer.  """
        e = self.get_existing_status(name)
        if not e:
            print(f"Entity {name} not found")
        else:
            self._reset(e, except_list)

    def _reset(self, e, except_list=[]):
        if self.is_locked_by_other(e):
            node = self.get_lock(e)
            print(f"Skipping {e['RowKey']} as it is locked by {node}")
        elif self._reset_metrics(e, except_list):
            e['status'] = 'reset'
            print(f"Resetting entity {e['RowKey']}")
            self.update_status_entity(e)

    def reset_all(self, name):
        """ This resets all properties on all entities that are not locked by another.  """
        for e in self.get_all_status_entities():
            self._reset(e)

    def _reset_metrics(self, entity, except_list=[]):
        # now clear all data to force a full re-run of everything.
        modified = False
        for key in list(entity.keys()):
            if key != 'PartitionKey' and key != 'RowKey' and key != 'name' and key != 'status' and key != 'node' and key not in except_list:
                del entity[key]
                modified = True
        return modified

    def upload(self, name, path, reset, priority=0, **kwargs):
        """ Upload a file to the named folder in the blob store associated with this ArchaiStore and
        add the given named status row in our status table.  It also locks the row with 'uploading'
        status until the upload is complete which ensures another machine does not try
        processing work until the upload is finished. The path points to a file or a folder.
        If a folder it uploads everything in that folder. This can also optionally reset
        the row, since sometimes you want to upload a new model for training, then reset
        all the metrics computed on the previous model. The optional priority is just a
        added as a property on the row which can be used by a distributed job scheduler to
        prioritize the work that is being queued up in this table. """
        if not name:
            raise Exception('Entity name is missing')

        if '/' in name:
            raise Exception('Entity name cannot contain a slash')
        e = self.get_status(name)

        e = self.lock(name, 'uploading')
        if not e:
            return
        e['priority'] = priority
        self.merge_status_entity(e)

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
        if reset:
            self._reset_metrics(e)
        e['status'] = 'uploaded'
        for k in kwargs:
            e[k] = kwargs[k]

        self.unlock_entity(e)

    def batch_upload(self, path, glob_pattern='*.onnx', override=False, reset=False, priority=0, **kwargs):
        """ Upload all the matching files in the given path to the blob store
        where the status table 'name' will be the base name of the files found
        by the given non-recursive glob_pattern.
        """
        if not os.path.isdir(path):
            raise Exception(f'Path is not a directory: {path}')

        models = glob.glob(os.path.join(path, glob_pattern))
        if len(models) == 0:
            print(f"No *.onnx models found in {path}")

        for file in models:
            name = os.path.splitext(os.path.basename(file))[0]
            if override or not self.get_existing_status(name):
                self.upload(name, file, reset, priority, **kwargs)
            else:
                print(f"Skipping {name} as it already exists")

    def download(self, name, folder, specific_file=None):
        """ Download files from the given folder name from our associated blob container
        and return a list of the local paths to all downloaded files.  If an optional specific_file is
        given then it tries to find and download that file only.  Returns a list of local files created.
        The specific_file can be a regular expression like '*.onnx'. """
        container = self._get_container_client(self.blob_container_name)
        if not container.exists():
            return []

        if not os.path.isdir(folder):
            os.makedirs(folder)
        local_file = None
        prefix = f'{name}/'
        downloaded = []
        if specific_file:
            specific_file_re = re.compile(specific_file)

        for blob in container.list_blobs(name_starts_with=prefix):
            file_name = blob.name[len(prefix):]
            download = False
            if specific_file:
                if not specific_file_re.match(file_name):
                    continue
                else:
                    download = True
                    local_file = os.path.join(folder, file_name)
            else:
                download = True
                local_file = os.path.join(folder, file_name)

            if download:
                local_file = os.path.realpath(local_file)
                dir = os.path.dirname(local_file)
                if os.path.isfile(dir):
                    os.unlink(dir)
                elif os.path.isdir(local_file):
                    rmtree(local_file)
                os.makedirs(dir, exist_ok=True)
                blob_client = container.get_blob_client(blob)
                try:
                    with open(local_file, 'wb') as f:
                        data = blob_client.download_blob()
                        f.write(data.readall())
                    downloaded += [local_file]
                except Exception as e:
                    print(f"### Error downloading blob '{blob}' to local file: {e}")

        return downloaded

    def delete_blobs(self, name, specific_file=None):
        """ Delete all the blobs associated with the given entity name. """
        container = self._get_container_client(self.blob_container_name)
        prefix = f'{name}/'
        for blob in container.list_blobs(name_starts_with=prefix):
            file_name = blob.name[len(prefix):]
            if specific_file and file_name != specific_file:
                continue
            container.delete_blob(blob)

    def list_blobs(self, prefix=None):
        """ List all the blobs associated with the given prefix. """
        container = self._get_container_client(self.blob_container_name)
        return [blob.name for blob in container.list_blobs(name_starts_with=prefix)]

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
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
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
    parser = argparse.ArgumentParser(description='Upload a named model (and optional accompanying files) to azure blob store')
    parser.add_argument('name', help='Friendly name of the folder to put this in.')
    parser.add_argument('file', help='Path to the file to upload to Azure ' +
                        'or a folder to upload all files in that folder to the same azure blob folder.')
    parser.add_argument('--priority', type=int, help='Optional priority override for this job. ' +
                        'Larger numbers mean lower priority')
    parser.add_argument('--reset', help='Reset stats for the model if it exists already.', action="store_true")
    args = parser.parse_args(args)
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    store.upload(args.name, args.file, args.reset, priority=args.priority)


def batch_upload(con_str, args):
    parser = argparse.ArgumentParser(description='Upload a a set of *.onnx files to the azure blob store' +
                                     'using the file name of each onnx file as the friendly folder name in the store.')
    parser.add_argument('path', help='Path to the folder containing onnx files to upload to Azure ')
    parser.add_argument('--override', help='Allow overriding existing models in the store.', action="store_true")
    parser.add_argument('--reset', help='Reset stats for any models we are overriding.', action="store_true")
    parser.add_argument('--priority', type=int, help='Optional priority override for these jobs. ' +
                        'Larger numbers mean lower priority')
    args = parser.parse_args(args)
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    store.batch_upload(args.path, args.override, args.reset, priority=args.priority)


def download(con_str, args):
    parser = argparse.ArgumentParser(
        description="Download assets from azure blob store using friendly name.")
    parser.add_argument('--name', help='Friendly name of model to download (if not provided it downloads them all')
    parser.add_argument('--file', help='The optional name of the files to download instead of getting them all.')
    args = parser.parse_args(args)

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    friendly_name = args.name
    if not friendly_name:
        friendly_names = [e['name'] for e in store.get_all_status_entities()]
    else:
        friendly_names = [friendly_name]

    specific_file = args.file

    for friendly_name in friendly_names:
        downloaded = store.download(friendly_name, friendly_name, specific_file)
        if len(downloaded) == 0 and specific_file:
            print(f"file {specific_file} not found")


def delete(con_str, args):
    parser = argparse.ArgumentParser(description='Delete a model from azure using its friendly name')
    parser.add_argument('name', help='The friendly name allocated by the upload script.')
    parser.add_argument('--file', help='Delete just the one file associated with the friendly name.')
    args = parser.parse_args(args)

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    store.delete_blobs(args.name, args.file)
    if not args.file:
        store.delete_status(args.name)


def reset(con_str, args):
    parser = argparse.ArgumentParser(
        description='Reset the named entity.')
    parser.add_argument('name', help='The friendly name to reset or "*" to reset all rows', default=None)
    args = parser.parse_args(args)
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    if args.name == "*":
        store.reset_all()
    else:
        store.reset(args.name)


def unlock(con_str, args):
    parser = argparse.ArgumentParser(
        description='Unlock all jobs for given node or unlock all jobs.')
    parser.add_argument('--node', help='Optional node name (default None).')
    args = parser.parse_args(args)
    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    store.unlock_all(args.node)


if __name__ == '__main__':
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    if len(sys.argv) <= 1:
        print("Expecting a command, one of 'status', 'upload', 'batch_upload', 'delete', 'download', 'reset', 'unlock'")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == 'status':
        status(con_str, args)
    elif cmd == 'upload':
        upload(con_str, args)
    elif cmd == 'batch_upload':
        batch_upload(con_str, args)
    elif cmd == 'download':
        download(con_str, args)
    elif cmd == 'delete':
        delete(con_str, args)
    elif cmd == 'reset':
        reset(con_str, args)
    elif cmd == 'unlock':
        unlock(con_str, args)
    else:
        print(f"Unknown command: {cmd}, expecting one of status, upload, download, delete, lock, unlock")
        sys.exit(1)
