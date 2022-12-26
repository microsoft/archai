import tensorflow as tf
import json, base64
import numpy as np
#import model_metrics_pb2
import pickle

from archai.common import utils

dataset_file = utils.full_path('~/dataroot/nasbench_ds/nasbench_only108.tfrecord')

records = []
for serialized_row in tf.python_io.tf_record_iterator(dataset_file):
      module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (
          json.loads(serialized_row.decode('utf-8')))
    #   dim = int(np.sqrt(len(raw_adjacency)))
    #   adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
    #   adjacency = np.reshape(adjacency, (dim, dim))
    #   operations = raw_operations.split(',')
    #   metrics = base64.b64decode(raw_metrics)
      records.append((module_hash, epochs, raw_adjacency, raw_operations, raw_metrics))

with open(dataset_file + '.pkl', 'wb') as f:
    pickle.dump(records, f)