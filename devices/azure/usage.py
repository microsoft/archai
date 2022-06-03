# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import uuid
import os
import sys
import re
import logging
import datetime
from azure.data.tables import TableServiceClient, UpdateMode, EntityProperty, EdmType


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'
USAGE_TABLE_NAME = 'USAGE_TABLE_NAME'

USAGE_TABLE = 'usage'
CONNECTION_STRING = ''


def get_utc_date():
    current_date = datetime.datetime.now()
    current_date = current_date.replace(tzinfo=datetime.timezone.utc)
    return current_date.isoformat()


def validate_table_name(name):
    x = re.search("^[A-Za-z][A-Za-z0-9]{2,62}$/", name)
    if not x:
        print(f"Table names cannot start with digits and must be from 3 to 63 alpha numeric chars: {name}")
        sys.exit(1)


def get_connection_string():
    global CONNECTION_STRING
    if not CONNECTION_STRING:
        CONNECTION_STRING = os.getenv(CONNECTION_NAME)
        if not CONNECTION_STRING:
            print(f"Please specify your {CONNECTION_NAME} environment variable.")
            sys.exit(1)
        st = os.getenv('USAGE_TABLE_NAME')
        if st:
            USAGE_TABLE = st.strip()
            validate_table_name(USAGE_TABLE)
    return CONNECTION_STRING


def get_usage_table_service():
    conn_str = get_connection_string()
    logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    logger.setLevel(logging.ERROR)
    return TableServiceClient.from_connection_string(conn_str=conn_str, logger=logger, logging_enable=False)


def get_all_usage_entities(name_filter=None, service=None):
    """ Get all usage entities with optional device name filter """
    global USAGE_TABLE
    if not service:
        service = get_usage_table_service()
    table_client = service.create_table_if_not_exists(USAGE_TABLE)

    entities = []
    query = "PartitionKey eq 'main'"
    if name_filter:
        query += f" and name eq '{name_filter}'"

    try:
        for e in table_client.query_entities(query_filter=query):
            entities += [e]

    except Exception as e:
        print(f"### error reading table: {e}")

    return entities


def update_usage_entity(entity, service=None):
    global USAGE_TABLE
    if not service:
        service = get_usage_table_service()
    table_client = service.create_table_if_not_exists(USAGE_TABLE)
    table_client.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)


def add_usage(name, start, end, service=None):
    global USAGE_TABLE
    if not service:
        service = get_usage_table_service()
    table_client = service.create_table_if_not_exists(USAGE_TABLE)

    entity = {
        'PartitionKey': 'main',
        'RowKey': str(uuid.uuid4()),
        'name': name,
        'start': start,
        'end': end
    }

    update_usage_entity(entity, service=service)
    return entity


def print_usage_entities(entities):
    keys = []
    for e in entities:
        for k in e:
            if k not in keys and k != 'PartitionKey' and k != 'RowKey':
                keys += [k]

    # so we can convert to .csv format
    print(", ".join(keys))
    for e in entities:
        for k in keys:
            if k in e:
                x = e[k]
                if isinstance(x, EntityProperty) and x.edm_type is EdmType.INT64:
                    x = x.value
                v = str(x).replace(',', ' ')
                print(f"{v}", end='')
            print(', ', end='')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print usage in .csv format using ' +
        f'{CONNECTION_NAME} environment variable.')
    parser.add_argument('--usage', help='Optional match for the name column (default None).')
    args = parser.parse_args()
    entities = get_all_usage_entities(args.usage)
    print_usage_entities(entities)
