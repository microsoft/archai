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
        # get total number of ops incoming to this node
        in_ops = [(edge,op) for edge in node for op in edge._op.ops()]
        assert len(in_ops) >= max_final_edges

        selected = random.sample(in_ops, max_final_edges)
        # finalize selected op, select 1st value from return which is op finalized desc
        selected_edges = [EdgeDesc(s[1].finalize()[0], s[0].input_ids) \
                            for s in selected]
        return NodeDesc(selected_edges)
