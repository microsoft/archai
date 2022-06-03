import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path += [os.path.join(SCRIPT_DIR, '..', 'qualcomm')]

from dlc_helper import get_dlc_metrics
from status import get_all_status_entities, update_status_entity
from download import download_model
from azure.data.tables import EntityProperty, EdmType

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def get_dlc_html(entity, conn_string):
    name = entity['name']
    temp = os.getenv('TEMP')
    model_found, _, local_file = download_model(name, temp, conn_string, specific_file='model.quant.html')
    if model_found:
        return local_file
    return None


def update_total_params(conn_string):
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    for e in get_all_status_entities():
        name = e['name']
        if 'params' not in e:
            html = get_dlc_html(e, conn_string)
            if html:
                _, params = get_dlc_metrics(html)
                if params != 0:
                    print(f"model {name} has {params} parameters")
                    e['params'] = EntityProperty(params, EdmType.INT64)
                    update_status_entity(e)


if __name__ == '__main__':    
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        print(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    update_total_params(conn_string)