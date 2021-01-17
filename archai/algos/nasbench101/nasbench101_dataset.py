# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""User interface for the NAS Benchmark dataset.

Before using this API, download the data files from the links in the README.

Usage:
  # Load the data from file (this will take some time)
  nasbench = api.NASBench('/path/to/nasbench.tfrecord')

  # Create an Inception-like module (5x5 convolution replaced with two 3x3
  # convolutions).
  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]],   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])


  # Query this model from dataset
  data = nasbench.query(model_spec)

Adjacency matrices are expected to be upper-triangular 0-1 matrices within the
defined search space (7 vertices, 9 edges, 3 allowed ops). The first and last
operations must be 'input' and 'output'. The other operations should be from
config['available_ops']. Currently, the available operations are:
  CONV3X3 = "conv3x3-bn-relu"
  CONV1X1 = "conv1x1-bn-relu"
  MAXPOOL3X3 = "maxpool3x3"

When querying a spec, the spec will first be automatically pruned (removing
unused vertices and edges along with ops). If the pruned spec is still out of
the search space, an OutOfDomainError will be raised, otherwise the data is
returned.

The returned data object is a dictionary with the following keys:
  - module_adjacency: numpy array for the adjacency matrix
  - module_operations: list of operation labels
  - trainable_parameters: number of trainable parameters in the model
  - training_time: the total training time in seconds up to this point
  - train_accuracy: training accuracy
  - validation_accuracy: validation_accuracy
  - test_accuracy: testing accuracy

Instead of querying the dataset for a single run of a model, it is also possible
to retrieve all metrics for a given spec, using:

  fixed_stats, computed_stats = nasbench.get_metrics_from_spec(model_spec)

The fixed_stats is a dictionary with the keys:
  - module_adjacency
  - module_operations
  - trainable_parameters

The computed_stats is a dictionary from epoch count to a list of metric
dicts. For example, computed_stats[108][0] contains the metrics for the first
repeat of the provided model trained to 108 epochs. The available keys are:
  - halfway_training_time
  - halfway_train_accuracy
  - halfway_validation_accuracy
  - halfway_test_accuracy
  - final_training_time
  - final_train_accuracy
  - final_validation_accuracy
  - final_test_accuracy
"""

from typing import Dict, List, Optional, Tuple
import base64
import copy
import json
import os
import random
import time
import pickle
import logging

import numpy as np
from numpy.lib.function_base import average

from torch import nn

from archai.common import utils
from . import config
from . import model_metrics_pb2
from . import model_spec as _model_spec
from . import model_builder

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec

class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""


class Nasbench101Dataset(object):
  """User-facing API for accessing the NASBench dataset."""

  def __init__(self, dataset_file, seed=None):
    """Initialize dataset, this should only be done once per experiment.

    Args:
      dataset_file: path to .tfrecord file containing the dataset.
      seed: random seed used for sampling queried models. Two NASBench objects
        created with the same seed will return the same data points when queried
        with the same models in the same order. By default, the seed is randomly
        generated.
    """
    self.config = config.build_config()
    random.seed(seed)

    dataset_file = utils.full_path(dataset_file)
    logging.info(f'Loading dataset from file "{dataset_file}"...')
    start = time.time()

    # Stores the fixed statistics that are independent of evaluation (i.e.,
    # adjacency matrix, operations, and number of parameters).
    # hash --> metric name --> scalar
    self.fixed_statistics:Dict[str, dict] = {}

    # Stores the statistics that are computed via training and evaluating the
    # model on CIFAR-10. Statistics are computed for multiple repeats of each
    # model at each max epoch length.
    # hash --> epochs --> repeat index --> metric name --> scalar
    self.computed_statistics:Dict[str, Dict[int, List[dict]]] = {}

    # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
    # {108} for the smaller dataset with only the 108 epochs.
    self.valid_epochs = set()

    with open(dataset_file, 'rb') as f:
      records = pickle.load(f)

    for module_hash, epochs, raw_adjacency, raw_operations, raw_metrics in records:
      dim = int(np.sqrt(len(raw_adjacency)))
      adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
      adjacency = np.reshape(adjacency, (dim, dim))
      operations = raw_operations.split(',')
      metrics = model_metrics_pb2.ModelMetrics.FromString(
          base64.b64decode(raw_metrics))

      if module_hash not in self.fixed_statistics:
        # First time seeing this module, initialize fixed statistics.
        new_entry = {}
        new_entry['module_adjacency'] = adjacency
        new_entry['module_operations'] = operations
        new_entry['trainable_parameters'] = metrics.trainable_parameters
        self.fixed_statistics[module_hash] = new_entry
        self.computed_statistics[module_hash] = {}

      self.valid_epochs.add(epochs)

      if epochs not in self.computed_statistics[module_hash]:
        self.computed_statistics[module_hash][epochs] = []

      # Each data_point consists of the metrics recorded from a single
      # train-and-evaluation of a model at a specific epoch length.
      data_point = {}

      # Note: metrics.evaluation_data[0] contains the computed metrics at the
      # start of training (step 0) but this is unused by this API.

      # Evaluation statistics at the half-way point of training
      half_evaluation = metrics.evaluation_data[1]
      data_point['halfway_training_time'] = half_evaluation.training_time
      data_point['halfway_train_accuracy'] = half_evaluation.train_accuracy
      data_point['halfway_validation_accuracy'] = (
          half_evaluation.validation_accuracy)
      data_point['halfway_test_accuracy'] = half_evaluation.test_accuracy

      # Evaluation statistics at the end of training
      final_evaluation = metrics.evaluation_data[2]
      data_point['final_training_time'] = final_evaluation.training_time
      data_point['final_train_accuracy'] = final_evaluation.train_accuracy
      data_point['final_validation_accuracy'] = (
          final_evaluation.validation_accuracy)
      data_point['final_test_accuracy'] = final_evaluation.test_accuracy

      self.computed_statistics[module_hash][epochs].append(data_point)

    module_hash_acc:List[Tuple[float, str]] = []
    for module_hash, fixed_stat in self.fixed_statistics.items():
      computed_epochs_stat = self.computed_statistics[module_hash]
      max_epochs = max(computed_epochs_stat.keys())
      fixed_stat['max_epochs'] = max_epochs
      computed_stats = computed_epochs_stat[max_epochs]

      fixed_stat['all_final_training_time'] = [c['final_training_time'] for c in computed_stats]
      fixed_stat['all_final_train_accuracy'] = [c['final_train_accuracy'] for c in computed_stats]
      fixed_stat['all_final_validation_accuracy'] = [c['final_validation_accuracy'] for c in computed_stats]
      fixed_stat['all_final_test_accuracy'] = [c['final_test_accuracy'] for c in computed_stats]
      fixed_stat['avg_final_test_accuracy'] = average(fixed_stat['all_final_test_accuracy'])

      module_hash_acc.append((fixed_stat['avg_final_test_accuracy'], module_hash))

    self.module_hashes = sorted(module_hash_acc, key=lambda v: v[0])
    for i, (acc, module_hash) in enumerate(self.module_hashes):
      self.fixed_statistics[module_hash]['index'] = i

    elapsed = time.time() - start
    logging.info('Loaded dataset in %d seconds' % elapsed)

  def __len__(self):
      return len(self.module_hashes)

  def __getitem__(self, idx, epochs=108, stop_halfway=False,
                  run_index:Optional[int]=None):
    module_hash = self.module_hashes[idx][1]
    fixed_stat, computed_stat = self.get_metrics_from_hash(module_hash)
    return self._join_fixed_computed(fixed_stat, computed_stat, epochs=epochs,
                                     stop_halfway=stop_halfway,
                                     run_index=run_index)

  def create_model_spec(self, desc_matrix:List[List[int]], vertex_ops:List[str])->ModelSpec:
    return ModelSpec(desc_matrix, vertex_ops)

  def query(self, desc_matrix:List[List[int]], vertex_ops:List[str],
            epochs=108, stop_halfway=False, run_index:Optional[int]=None):
    """Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function also allows querying the evaluation metrics at the halfway
    point of training using stop_halfway. Using this option will increment the
    budget counters only up to the halfway point.

    Args:
      desc_matrix: matric describing edges
      vertex_ops: list of ops for cell edges
      epochs: number of epochs trained. Must be one of the evaluated number of
        epochs, [4, 12, 36, 108] for the full dataset.
      stop_halfway: if True, returned dict will only contain the training time
        and accuracies at the halfway point of training (num_epochs/2).
        Otherwise, returns the time and accuracies at the end of training
        (num_epochs).
      run_index: index of run. If None then return random run.

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """

    model_spec = self.create_model_spec(desc_matrix, vertex_ops)

    if epochs not in self.valid_epochs:
      raise OutOfDomainError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)

    fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
    return self._join_fixed_computed(fixed_stat, computed_stat, epochs=epochs,
                                     stop_halfway=stop_halfway,
                                     run_index=run_index)


  def _join_fixed_computed(self, fixed_stat, computed_stat,
                           epochs=108, stop_halfway=False,
                           run_index:Optional[int]=None)->dict:
    if run_index is None:
      run_index = random.randint(0, self.config['num_repeats'] - 1)

    computed_stat = computed_stat[epochs][run_index]

    data = {'run_index':run_index, 'epochs':epochs, 'stop_halfway':stop_halfway}
    data['module_adjacency'] = fixed_stat['module_adjacency']
    data['module_operations'] = fixed_stat['module_operations']
    data['trainable_parameters'] = fixed_stat['trainable_parameters']
    data['index'] = fixed_stat['index']

    data['all_final_training_time'] = fixed_stat['all_final_training_time']
    data['all_final_train_accuracy'] = fixed_stat['all_final_train_accuracy']
    data['all_final_validation_accuracy'] = fixed_stat['all_final_validation_accuracy']
    data['all_final_test_accuracy'] = fixed_stat['all_final_test_accuracy']
    data['avg_final_test_accuracy'] = fixed_stat['avg_final_test_accuracy']

    if stop_halfway:
      data['training_time'] = computed_stat['halfway_training_time']
      data['train_accuracy'] = computed_stat['halfway_train_accuracy']
      data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
      data['test_accuracy'] = computed_stat['halfway_test_accuracy']
    else:
      data['training_time'] = computed_stat['final_training_time']
      data['train_accuracy'] = computed_stat['final_train_accuracy']
      data['validation_accuracy'] = computed_stat['final_validation_accuracy']
      data['test_accuracy'] = computed_stat['final_test_accuracy']

    return data

  def create_model(self, index:int, device=None,
          stem_out_channels=128, num_stacks=3, num_modules_per_stack=3, num_labels=10)->nn.Module:
    data = self[index]
    adj, ops = data['module_adjacency'], data['module_operations']
    return model_builder.build(adj, ops, device=device,
          stem_out_channels=stem_out_channels, num_stacks=num_stacks,
          num_modules_per_stack=num_modules_per_stack, num_labels=num_labels)

  def is_valid(self, desc_matrix:List[List[int]], vertex_ops:List[str]):
    """Checks the validity of the model_spec.

    For the purposes of benchmarking, this does not increment the budget
    counters.

    Returns:
      True if model is within space.
    """

    model_spec = self.create_model_spec(desc_matrix, vertex_ops)

    try:
      self._check_spec(model_spec)
    except OutOfDomainError:
      return False

    return True

  def get_metrics_from_hash(self, module_hash):
    """Returns the metrics for all epochs and all repeats of a hash.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      module_hash: MD5 hash from the key

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    fixed_stat = self.fixed_statistics[module_hash]
    computed_stat = self.computed_statistics[module_hash]
    return fixed_stat, computed_stat

  def get_metrics_from_spec(self, model_spec):
    """Returns the metrics for all epochs and all repeats of a model.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    self._check_spec(model_spec)
    module_hash = self._hash_spec(model_spec)
    return self.get_metrics_from_hash(module_hash)

  def _check_spec(self, model_spec):
    """Checks that the model spec is within the dataset."""
    if not model_spec.valid_spec:
      raise OutOfDomainError('invalid spec, provided graph is disconnected.')

    num_vertices = len(model_spec.ops)
    num_edges = np.sum(model_spec.matrix)

    if num_vertices > self.config['module_vertices']:
      raise OutOfDomainError('too many vertices, got %d (max vertices = %d)'
                             % (num_vertices, self.config['module_vertices']))

    if num_edges > self.config['max_edges']:
      raise OutOfDomainError('too many edges, got %d (max edges = %d)'
                             % (num_edges, self.config['max_edges']))

    if model_spec.ops[0] != 'input':
      raise OutOfDomainError('first operation should be \'input\'')
    if model_spec.ops[-1] != 'output':
      raise OutOfDomainError('last operation should be \'output\'')
    for op in model_spec.ops[1:-1]:
      if op not in self.config['available_ops']:
        raise OutOfDomainError('unsupported op %s (available ops = %s)'
                               % (op, self.config['available_ops']))

  def _hash_spec(self, model_spec):
    """Returns the MD5 hash for a provided model_spec."""
    return model_spec.hash_spec(self.config['available_ops'])


class _NumpyEncoder(json.JSONEncoder):
  """Converts numpy objects to JSON-serializable format."""

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      # Matrices converted to nested lists
      return obj.tolist()
    elif isinstance(obj, np.generic):
      # Scalars converted to closest Python type
      return np.asscalar(obj)
    return json.JSONEncoder.default(self, obj)

