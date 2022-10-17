from typing import List, Tuple, Optional, Iterator, Dict
from overrides import overrides

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

from archai.common.common import get_conf
from archai.common.common import get_expdir
from archai.common.common import logger
from archai.datasets.data import get_data
from archai.nas.model import Model
from archai.nas.cell import Cell
from archai.nas.model_desc import CellDesc, ModelDesc, NodeDesc, EdgeDesc
from archai.nas.finalizers import Finalizers
from archai.algos.gumbelsoftmax.gs_op import GsOp


class GsFinalizers(Finalizers):

    @overrides
    def finalize_node(self, node:nn.ModuleList, node_index:int,
                      node_desc:NodeDesc, max_final_edges:int,
                      *args, **kwargs)->NodeDesc:
        conf = get_conf()
        gs_num_sample = conf['nas']['search']['model_desc']['cell']['gs']['num_sample']

        # gather the alphas of all edges in this node
        node_alphas = []
        for edge in node:
            if hasattr(edge._op, 'PRIMITIVES') and type(edge._op) == GsOp:
                alphas = [alpha for op, alpha in edge._op.ops()]
                node_alphas.extend(alphas)

        # TODO: will creating a tensor from a list of tensors preserve the graph?
        node_alphas = torch.Tensor(node_alphas)

        assert node_alphas.nelement() > 0

        # sample ops via gumbel softmax
        sample_storage = []
        for _ in range(gs_num_sample):
            sampled = F.gumbel_softmax(node_alphas, tau=1, hard=True, eps=1e-10, dim=-1)
            sample_storage.append(sampled)

        samples_summed = torch.sum(torch.stack(sample_storage, dim=0), dim=0)

        # send the sampled op weights to their
        # respective edges to be used for edge level finalize
        selected_edges = []
        counter = 0
        for _, edge in enumerate(node):
            if hasattr(edge._op, 'PRIMITIVES') and type(edge._op) == GsOp:
                this_edge_sampled_weights = samples_summed[counter:counter+len(edge._op.PRIMITIVES)]
                counter += len(edge._op.PRIMITIVES)
                # finalize the edge
                if this_edge_sampled_weights.bool().any():
                    op_desc, _ = edge._op.finalize(this_edge_sampled_weights)
                    new_edge = EdgeDesc(op_desc, edge.input_ids)
                    selected_edges.append(new_edge)

        # delete excess edges
        if len(selected_edges) > max_final_edges:
            # since these are sample edges there is no ordering
            # amongst them so we just arbitrarily select a few
            selected_edges = selected_edges[:max_final_edges]

        return NodeDesc(selected_edges, node_desc.conv_params)





