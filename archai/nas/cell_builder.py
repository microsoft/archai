# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC

from overrides import EnforceOverrides

from .model_desc import ModelDesc, NodeDesc

class ModelDescBuilder(ABC, EnforceOverrides):
    """This is interface class for different NAS algorithms to implement"""

    def register_ops(self)->None:
        pass

    def build(self, model_desc:ModelDesc, search_iter:int)->None:
        pass

    def seed(self, model_desc:ModelDesc)->None:
        # prepare model as seed model before search iterations starts
        pass

    def add_node(self, model_desc:ModelDesc)->None:
        """Utility method to add empty node in each cell"""
        for cell_desc in model_desc.cell_descs():
            # new node requires reset because post op must recompute channels
            new_nodes = [n.clone() for n in cell_desc.nodes()]
            new_nodes.append(NodeDesc(edges=[]))
            cell_desc.reset_nodes(new_nodes, cell_desc.node_shapes,
                              cell_desc.post_op, cell_desc.out_shape)

