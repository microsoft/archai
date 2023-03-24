import os
import numpy as np
import tempfile
import uuid
from archai.common.store import ArchaiStore

CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'


def test_store():
    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        print(f"Skipping test_store because you have no {CONNECTION_NAME} environment variable.")
        return

    storage_account_name, storage_account_key = ArchaiStore.parse_connection_string(con_str)
    store = ArchaiStore(storage_account_name, storage_account_key)
    name = str(uuid.uuid4())
    try:
        entities = store.get_all_status_entities()
        assert len([x for x in entities if x['name'] == name]) == 0

        e = store.get_status(name)
        assert e['status'] == 'new'
        e['status'] = 'running'
        e['accuracy'] = np.array([1.234])[0]  # test np.float
        e['params'] = 9223372036854775800
        store.update_status_entity(e)

        e = store.get_status(name)
        assert e['status'] == 'running'
        assert e['accuracy'] == 1.234
        assert e['params'] == 9223372036854775800
        entities = store.get_all_status_entities('status', 'running')
        assert len([x for x in entities if x['name'] == name]) == 1

        store.delete_status_entity(e)
        entities = store.get_all_status_entities()
        assert len([x for x in entities if x['name'] == name]) == 0

        store.lock(name, 'uploading')
        assert store.is_locked(name)
        assert store.is_locked_by_self(name)

        e = store.get_status(name)
        assert e['status'] == 'uploading'

        path = os.path.realpath(__file__)
        filename = os.path.basename(path)
        store.upload_blob(name, path)

        store.unlock(name)
        assert not store.is_locked(name)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.download(name, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, filename))

    finally:
        store.delete_blobs(name)
        store.delete_status_entity(e)
