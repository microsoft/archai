# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import time
from runner import lock_job, unlock_job, get_unique_node_id, set_unique_node_id
from status import update_status, get_status_table_service, get_status, get_connection_string


def get_lock(entity):
    return entity['node'] if 'node' in entity else 'none'


def test(entity, delay, service):
    print(f"### locking entity: {entity['name']}")
    try:
        lock_job(entity, service)
        e = get_status(entity['name'], service=service)
        print(f"==========> locked by: {get_lock(e)} ========================")
        time.sleep(5)
    except Exception as ex:
        print(f'{ex}')
    time.sleep(delay)
    print(f"### unlocking entity: {entity['name']}")
    try:
        unlock_job(entity, service)
    except Exception as ex:
        print(f'{ex}')

    e = get_status(entity['name'], service=service)
    print(f"### entity locked by: {get_lock(e)}")
    time.sleep(delay)
    return e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unit test for the lock/unlock methods on our Azure status table.')
    parser.add_argument('--name', help='Name of entity to use.')
    parser.add_argument('--delay', type=float, help='Seconds delay between lock unlock ops (default 1).', default=1)
    args = parser.parse_args()

    set_unique_node_id(get_unique_node_id() + f'_{os.getpid()}')
    name = args.name
    delay = args.delay
    service = get_status_table_service(get_connection_string())
    entity = update_status(name, 'testing', service=service)

    count = 100
    for i in range(count):
        print(f'======= test {i} of {count} ============================================')
        entity = test(entity, delay, service)
