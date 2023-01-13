# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides
import random

from torch import nn

from archai.nas.model_desc import NodeDesc, EdgeDesc
from archai.nas.finalizers import Finalizers
from archai.nas.operations import Zero


class RandomFinalizers(Finalizers):
    @overrides
    def finalize_node(self, node:nn.ModuleList, node_index:int,
                      node_desc:NodeDesc, max_final_edges:int,
                      *args, **kwargs)->NodeDesc:
        # get total number of ops incoming to this node
        in_ops = [(edge,op) for edge in node \
                            for op, order in edge._op.ops()
                            if not isinstance(op, Zero)]
        assert len(in_ops) >= max_final_edges

        selected = random.sample(in_ops, max_final_edges)
        # finalize selected op, select 1st value from return which is op finalized desc
        selected_edges = [EdgeDesc(s[1].finalize()[0], s[0].input_ids) \
                            for s in selected]
        return NodeDesc(selected_edges, node_desc.conv_params)
