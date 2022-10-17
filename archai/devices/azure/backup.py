import argparse
import sys
import tqdm
from status import get_all_status_entities, get_status_table_service, update_status_entity
from usage import get_all_usage_entities, get_usage_table_service, update_usage_entity

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'
STATUS_TABLE_NAME = 'STATUS_TABLE_NAME'

STATUS_TABLE = 'status'
CONNECTION_STRING = ''

# the blobs can be easily copied using azcopy, see https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-blobs-copy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Backup the status and usage tables to a new azure storage account ' +
        f'{CONNECTION_NAME} environment variable.')
    parser.add_argument('--target', help='Connection string for the target storage account.')
    args = parser.parse_args()
    if not args.target:
        print("Please provide --target connection string")
        sys.exit(1)

    entities = get_all_status_entities()

    target = get_status_table_service(args.target)

    print(f"Uploading {len(entities)} status entities...")
    # upload the entities to the new service.
    with tqdm.tqdm(total=len(entities)) as pbar:
        for e in entities:
            update_status_entity(e, target)
            pbar.update(1)

    usage = get_all_usage_entities()
    print(f"Uploading {len(usage)} usage entities...")
    target = get_usage_table_service(args.target)
    # upload the usage to the new service.
    with tqdm.tqdm(total=len(usage)) as pbar:
        for u in usage:
            update_usage_entity(u, target)
            pbar.update(1)
