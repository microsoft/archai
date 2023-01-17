# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, Iterable, List, Optional, Tuple, Dict
from abc import ABC, abstractmethod

from overrides import overrides, EnforceOverrides

import numpy as np

import torch
from torch import nn, tensor
from overrides import overrides, EnforceOverrides

import archai.algos.divnas.analyse_activations as aa

from archai.nas.cell import Cell
from archai.nas.operations import Zero
from archai.nas.operations import Op


class Divnas_Cell():
    ''' Wrapper cell class for divnas specific modifications '''
    def __init__(self, cell:Cell):

        self._cell = cell

        self._collect_activations = False
        self._edgeoptype = None
        self._sigma = None
        self._counter = 0
        self.node_covs:Dict[int, np.array] = {}
        self.node_num_to_node_op_to_cov_ind:Dict[int, Dict[Op, int]] = {}        
        
    def collect_activations(self, edgeoptype, sigma:float)->None:
        self._collect_activations = True
        self._edgeoptype = edgeoptype
        self._sigma = sigma

        # collect bookkeeping info
        for i, node in enumerate(self._cell.dag):
            node_op_to_cov_ind:Dict[Op, int] = {}
            counter = 0
            for edge in node:
                for op, alpha in edge._op.ops():
                    if isinstance(op, Zero):
                        continue
                    node_op_to_cov_ind[op] = counter
                    counter += 1                        
            self.node_num_to_node_op_to_cov_ind[i] = node_op_to_cov_ind


        # go through all edges in the DAG and if they are of edgeoptype
        # type then set them to collect activations
        for i, node in enumerate(self._cell.dag):            
            # initialize the covariance matrix for this node
            num_ops = 0
            for edge in node:
                if hasattr(edge._op, 'PRIMITIVES') and type(edge._op) == self._edgeoptype:
                    num_ops += edge._op.num_primitive_ops - 1
                    edge._op.collect_activations = True
                   
            self.node_covs[id(node)] = np.zeros((num_ops, num_ops))
            

    def update_covs(self):
        assert self._collect_activations

        for _, node in enumerate(self._cell.dag):
            # TODO: convert to explicit ordering
            all_activs = []
            for j, edge in enumerate(node):
                if type(edge._op) == self._edgeoptype:
                    activs = edge._op.activations
                    all_activs.append(activs)
            # update covariance matrix    
            activs_converted = self._convert_activations(all_activs)
            new_cov = aa.compute_rbf_kernel_covariance(activs_converted, sigma=self._sigma)
            updated_cov = (self._counter * self.node_covs[id(node)] + new_cov) / (self._counter + 1)
            self.node_covs[id(node)] = updated_cov


    def clear_collect_activations(self):
        for _, node in enumerate(self._cell.dag):            
            for edge in node:
                if hasattr(edge._op, 'PRIMITIVES') and type(edge._op) == self._edgeoptype:
                    edge._op.collect_activations = False

        self._collect_activations = False
        self._edgeoptype = None
        self._sigma = None
        self._node_covs = {}


    def _convert_activations(self, all_activs:List[List[np.array]])->List[np.array]:
        ''' Converts to the format needed by covariance computing functions
        Input all_activs: List[List[np.array]]. Outer list len is num_edges. 
        Inner list is of num_ops length. Each element in inner list is [batch_size, x, y, z] '''

        num_ops = len(all_activs[0])
        for activs in all_activs:
            assert num_ops == len(activs)

        all_edge_list = []
        
        for edge in all_activs:
            obsv_dict = defaultdict(list)
            # assumption edge_np will be (num_ops, batch_size, x, y, z)
            edge_np = np.array(edge)
            for op in range(edge_np.shape[0]):
                for b in range(edge_np.shape[1]):
                    feat = edge_np[op][b]
                    feat = feat.flatten()
                    obsv_dict[op].append(feat)

            feature_list = [*range(num_ops)]
            for key in obsv_dict.keys():
                feat = np.array(obsv_dict[key])
                feature_list[key] = feat

            all_edge_list.extend(feature_list)

        return all_edge_list
    