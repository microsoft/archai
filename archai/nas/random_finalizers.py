from typing import List, Tuple, Optional, Iterator, Dict, Set
from overrides import overrides
import random

import torch
from torch import nn

import numpy as np

from archai.common.common import get_conf
from archai.common.common import logger
from archai.datasets.data import get_data
from archai.nas.model import Model
from archai.nas.cell import Cell
from archai.nas.model_desc import CellDesc, ModelDesc, NodeDesc, EdgeDesc
from archai.nas.finalizers import Finalizers
from archai.algos.divnas.analyse_activations import compute_brute_force_sol
from archai.algos.divnas.divop import DivOp


class RandomFinalizers(Finalizers):


    @overrides
    def finalize_node(self, node:nn.ModuleList, max_final_edges:int)->NodeDesc:
        # node is a list of edges
        assert len(node) >= max_final_edges
        
        # get total number of ops incoming to this node
        num_ops = 0
        for edge in node:
            if hasattr(edge._op, 'PRIMITIVES'):
                num_ops += len(edge._op.PRIMITIVES)

        # and collect some bookkeeping indices
        edge_num_and_op_ind = []
        for j, edge in enumerate(node):
            if hasattr(edge._op, 'PRIMITIVES'):
                for k in range(len(edge._op.PRIMITIVES)):
                    edge_num_and_op_ind.append((j, k))

        assert len(edge_num_and_op_ind) == num_ops

        # run random subset selection
        rand_subset = self._random_subset(num_ops, max_final_edges)

        # convert the cov indices to edge descs
        selected_edges = []
        for ind in rand_subset:
            edge_ind, op_ind = edge_num_and_op_ind[ind]
            op_desc = node[edge_ind]._op.get_op_desc(op_ind)
            new_edge = EdgeDesc(op_desc, node[edge_ind].input_ids)
            selected_edges.append(new_edge)
                    
        return NodeDesc(selected_edges)


    def _random_subset(self, num_ops:int, max_final_edges:int)->Set[int]:
        assert num_ops > 0
        assert max_final_edges > 0
        assert max_final_edges <= num_ops

        S = set()
        while len(S) < max_final_edges:
            sample = random.randint(0, num_ops)
            S.add(sample)

        return S
