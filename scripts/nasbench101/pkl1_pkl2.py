from typing import Dict, OrderedDict
import pickle
import numpy as np
import base64
from collections import OrderedDict

from archai.supergraph.algos.nasbench101 import model_metrics_pb2
from archai.supergraph.utils import utils

def eval_to_dict(e)->dict:
    return {
        'training_time': e.training_time,
        'train_accuracy': e.train_accuracy,
        'validation_accuracy': e.validation_accuracy,
        'test_accuracy': e.test_accuracy
    }

def main():
    in_dataset_file = utils.full_path('~/dataroot/nasbench_ds/nasbench_full.tfrecord.pkl')
    out_dataset_file = utils.full_path('~/dataroot/nasbench_ds/nasbench_full.pkl')

    stats:Dict[str, dict] = {}

    with open(in_dataset_file, 'rb') as f:
        records = pickle.load(f)

    for module_hash, epochs, raw_adjacency, raw_operations, raw_metrics in records:
        dim = int(np.sqrt(len(raw_adjacency)))
        adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
        adjacency = np.reshape(adjacency, (dim, dim))
        operations = raw_operations.split(',')
        metrics = model_metrics_pb2.ModelMetrics.FromString(
            base64.b64decode(raw_metrics))

        if module_hash not in stats:
            stats[module_hash] = {
                'module_hash': module_hash,
                'module_adjacency': adjacency,
                'module_operations': operations,
                'trainable_parameters': metrics.trainable_parameters,
                'total_time': metrics.total_time,
                'metrics': {}
            }

        entry = stats[module_hash]
        assert entry['module_hash'] == module_hash
        #assert entry['module_adjacency'] == adjacency
        assert entry['module_operations'] == operations
        assert entry['trainable_parameters'] == metrics.trainable_parameters

        if epochs not in entry['metrics']:
            entry['metrics'][epochs] = []
        entry['metrics'][epochs].append([eval_to_dict(e) for e in metrics.evaluation_data])

    dataset = sorted(stats.values(), key=lambda d: np.mean([r[-1]['test_accuracy']  for r in d['metrics'][108]]))
    for i,d in enumerate(dataset):
        d['rank']=i

    odict = OrderedDict((d['module_hash'], d) for d in dataset)

    with open(out_dataset_file, 'wb') as f:
        pickle.dump(odict, f)

if __name__ == '__main__':
    main()