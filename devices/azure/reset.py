# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import tqdm
from status import get_existing_status, update_status_entity, get_all_status_entities, get_connection_string, get_status_table_service
from delete import delete_blobs

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def reset_metrics(entity, f1, ifs, macs):
    # now clear all data to force a full re-run of everything.
    if f1:
        for key in ['f1_1k', 'f1_10k', 'f1_1k_f', 'f1_onnx']:
            if key in entity:
                print(f"Resetting '{key}'")
                del entity[key]
    if ifs:
        if "mean" in entity:
            del entity["mean"]
        if "stdev" in entity:
            del entity["stdev"]
        if "total_inference_avg" in entity:
            del entity["total_inference_avg"]
    if macs and "macs" in entity:
        del entity["macs"]
    if macs and "params" in entity:
        del entity["params"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reset the status of the given model to "new" so it is re-tested, " +\
            f"using your {CONNECTION_NAME} environment variable.')
    parser.add_argument('name', help='Friendly name of model status to reset (or * to reset everything!)')
    parser.add_argument('--all', help='Reset all properties', action="store_true")
    parser.add_argument('--f1', help='Reset f1 score', action="store_true")
    parser.add_argument('--ifs', help='Reset total_inference_avg', action="store_true")
    parser.add_argument('--macs', help='Reset macs number', action="store_true")
    parser.add_argument('--quant', help='Reset quantized model (force requantization)', action="store_true")
    args = parser.parse_args()

    friendly_name = args.name
    all = args.all
    f1 = args.f1
    ifs = args.ifs
    macs = args.macs
    quant = args.quant

    if all:
        f1 = True
        ifs = True
        macs = True
        quant = True

    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    entities = []

    service = get_status_table_service(get_connection_string())

    if friendly_name == '*':
        a = input("Are you sure you want to reset everything (y o n)? ").strip().lower()
        if a != 'y' and a != 'yes':
            sys.exit(1)
        entities = get_all_status_entities()

    else:
        entity = get_existing_status(friendly_name)
        if not entity:
            print(f"Entity {friendly_name} not found")
            sys.exit(1)
        entities += [entity]

    with tqdm.tqdm(total=len(entities)) as pbar:
        for e in entities:
            name = e['name']
            reset_metrics(e, f1, ifs, macs)

            if quant:
                delete_blobs(name, 'model.dlc')
                delete_blobs(name, 'model.quant.dlc')

            e['status'] = 'reset'
            update_status_entity(e, service)
            pbar.update(1)
