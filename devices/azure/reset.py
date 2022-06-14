# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
from status import get_status, update_status_entity
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
        if "total_inference_avg" in entity:
            del entity["total_inference_avg"]
    if macs and "macs" in entity:
        del entity["macs"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reset the status of the given model to "new" so it is re-tested, " +\
            f"using your {CONNECTION_NAME} environment variable.')
    parser.add_argument('name', help='Friendly name of model status to reset')
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

    entity = get_status(friendly_name)
    reset_metrics(entity, f1, ifs, macs)
    entity['status'] = 'reset'
    update_status_entity(entity)

    if quant:
        delete_blobs(friendly_name, 'model.dlc')
        delete_blobs(friendly_name, 'model.quant.dlc')
