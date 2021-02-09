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



from typing import Dict, List, Optional, OrderedDict, Tuple
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

  VALID_EPOCHS = [4, 12, 36, 108]

  def __init__(self, dataset_file, seed=None):
    self.config = config.build_config()
    random.seed(seed)

    dataset_file = utils.full_path(dataset_file)
    logging.info(f'Loading dataset from file "{dataset_file}"...')
    start = time.time()

    with open(dataset_file, 'rb') as f:
      self.data:OrderedDict[str, dict] = pickle.load(f)
    self.module_hashes = list(self.data.keys())

    elapsed = time.time() - start
    logging.info('Loaded dataset in %d seconds' % elapsed)

  def __len__(self):
      return len(self.module_hashes)

  def __getitem__(self, idx):
    module_hash = self.module_hashes[idx]
    return self.data[module_hash]

  def get_data(self, idx, epochs:Optional[int]=108, run_index:Optional[int]=None,
                   step_index:Optional[int]=-1)->dict:
    module_hash = self.module_hashes[idx]
    d = self.data[module_hash]
    return self.filter_data(d, epochs=epochs, run_index=run_index, step_index=step_index)

  def filter_data(self, d:dict, epochs:Optional[int]=108, run_index:Optional[int]=None,
                   step_index:Optional[int]=-1)->dict:
    if epochs is not None:
      d = d['metrics'][epochs]
      if run_index is not None:
        d = d[run_index]
        if step_index is not None:
          d = d[step_index]
    return d

  def get_test_acc(self, idx, epochs=108, step_index=-1)->List[float]:
    module_hash = self.module_hashes[idx]
    runs = self.data[module_hash]['metrics'][epochs]
    return [r[step_index]['test_accuracy'] for r in runs]

  def create_model_spec(self, desc_matrix:List[List[int]], vertex_ops:List[str])->ModelSpec:
    return ModelSpec(desc_matrix, vertex_ops)

  def query(self, desc_matrix:List[List[int]], vertex_ops:List[str],
            epochs:Optional[int]=108, run_index:Optional[int]=None,
            step_index:Optional[int]=-1):
    model_spec = self.create_model_spec(desc_matrix, vertex_ops)

    d = self.get_metrics_from_spec(model_spec)
    return self.filter_data(d, epochs=epochs, run_index=run_index, step_index=step_index)

  def create_model(self, idx:int, device=None,
          stem_out_channels=128, num_stacks=3, num_modules_per_stack=3, num_labels=10)->nn.Module:
    module_hash = self.module_hashes[idx]
    d = self.data[module_hash]
    adj, ops = d['module_adjacency'], d['module_operations']
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

  def get_metrics_from_spec(self, model_spec):
    self._check_spec(model_spec)
    module_hash = self._hash_spec(model_spec)
    return self.data[module_hash]

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



