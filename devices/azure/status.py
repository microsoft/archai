# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import re
import logging
import datetime
from azure.data.tables import TableServiceClient, UpdateMode, EntityProperty, EdmType


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'
STATUS_TABLE_NAME = 'STATUS_TABLE_NAME'

STATUS_TABLE = 'status'
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
        st = os.getenv('STATUS_TABLE_NAME')
        if st:
            STATUS_TABLE = st.strip()
            validate_table_name(STATUS_TABLE)
    return CONNECTION_STRING


def get_status_table_service():
    conn_str = get_connection_string()
    logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    logger.setLevel(logging.ERROR)
    return TableServiceClient.from_connection_string(conn_str=conn_str, logger=logger, logging_enable=False)


def get_all_status_entities(status=None, not_equal=False, service=None):
    """ Get all status entities with optional status column filter.
    For example, pass "status=complete" to find all status rows that
    have the status "complete".  Pass not_equal of True if you want
    to check the status is not equal to the given value.
    """
    global STATUS_TABLE
    if not service:
        service = get_status_table_service()
    table_client = service.create_table_if_not_exists(STATUS_TABLE)

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


def get_status(name, service=None):
    global STATUS_TABLE
    if not service:
        service = get_status_table_service()
    table_client = service.create_table_if_not_exists(STATUS_TABLE)

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


def update_status_entity(entity, service=None):
    global STATUS_TABLE
    if not service:
        service = get_status_table_service()
    table_client = service.create_table_if_not_exists(STATUS_TABLE)
    table_client.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)


def merge_status_entity(entity, service=None):
    global STATUS_TABLE
    if not service:
        service = get_status_table_service()
    table_client = service.create_table_if_not_exists(STATUS_TABLE)
    table_client.update_entity(entity=entity, mode=UpdateMode.MERGE)


def update_status(name, status, priority=None, service=None):
    global STATUS_TABLE
    if not service:
        service = get_status_table_service()
    table_client = service.create_table_if_not_exists(STATUS_TABLE)

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
    update_status_entity(entity)
    return entity


def delete_status(name, service=None):
    global STATUS_TABLE
    if not service:
        service = get_status_table_service()
    table_client = service.create_table_if_not_exists(STATUS_TABLE)

    for e in get_all_status_entities():
        if 'name' in e and e['name'] == name:
            print(f"Deleting status entity for {name}")
            table_client.delete_entity(e)


def print_entities(entities, columns=None):
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
                v = str(x).replace(',', ' ')
                print(f"{v}", end='')
            print(', ', end='')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print status in .csv format using ' +
        f'{CONNECTION_NAME} environment variable.')
    parser.add_argument('--status', help='Optional match for the status column (default None).')
    parser.add_argument('--name', help='Optional name of single status row to return (default None).')
    parser.add_argument('--not_equal', '-ne', help='Switch the match to a not-equal comparison.', action="store_true")
    parser.add_argument('--locked', help='Find entities that are locked by a node.', action="store_true")
    parser.add_argument('--cols', help='Comma separated list of columns to report (default is to print all)')
    args = parser.parse_args()
    entities = get_all_status_entities(args.status, args.not_equal)
    if args.locked:
        entities = [e for e in entities if 'node' in e and e['node']]
    if args.name:
        entities = [e for e in entities if 'name' in e and e['name'] == args.name]

    columns = None
    if args.cols:
        columns = [x.strip() for x in args.cols.split(',')]
    print_entities(entities, columns)
