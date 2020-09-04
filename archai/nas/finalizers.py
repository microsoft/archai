# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.common.config import Config
from typing import List, Tuple, Optional, Iterator
from overrides import EnforceOverrides

from torch import nn

from archai.nas.model import Model
from archai.nas.cell import Cell
from archai.nas.model_desc import CellDesc, ModelDesc, NodeDesc, EdgeDesc

class Finalizers(EnforceOverrides):
    """Provides base algorithms for finalizing model, cell and edge which can be overriden

    For op-level finalize, just put logic in op's finalize.

    For model/cell/edge level finalize, you can override the methods in this class to customize the behavior. To override any of these methods, simply create new class in your algos folder, for example, diversity/diversity_finalizers.py. In this file create class that derives from Finalizers. Then in your algos exp_runner.py, return instance of that class in its finalizers() method.
    """

    def finalize_model(self, model:Model, to_cpu=True, restore_device=True)->ModelDesc:
        # move model to CPU before finalize because each op will serialize
        # its parameters and we don't want copy of these parameters hanging on GPU
        original = model.device_type()
        if to_cpu:
            model.cpu()

        # finalize will create copy of state and this can overflow GPU RAM
        assert model.device_type() == 'cpu'

        cell_descs = self.finalize_cells(model)

        if restore_device:
            model.to(original, non_blocking=True)

        return ModelDesc(conf_model_desc=model.desc.conf_model_desc,
                         model_stems=[op.finalize()[0] for op in model.model_stems],
                         pool_op=model.pool_op.finalize()[0],
                         cell_descs=cell_descs,
                         aux_tower_descs=model.desc.aux_tower_descs,
                         logits_op=model.logits_op.finalize()[0])

    def finalize_cells(self, model:Model)->List[CellDesc]:
        return [self.finalize_cell(cell, i, model.desc) \
                for i,cell in enumerate(model.cells)]

    def finalize_cell(self, cell:Cell, cell_index:int,
                      model_desc:ModelDesc, *args, **kwargs)->CellDesc:
        # first finalize each node, we will need to recreate node desc with final version
        max_final_edges = model_desc.max_final_edges

        node_descs:List[NodeDesc] = []
        for i,node in enumerate(cell.dag):
            node_desc = self.finalize_node(node, i, cell.desc.nodes()[i],max_final_edges)
            node_descs.append(node_desc)

        desc = cell.desc
        finalized = CellDesc(
            id = desc.id, cell_type=desc.cell_type, conf_cell=desc.conf_cell,
            stems=[cell.s0_op.finalize()[0], cell.s1_op.finalize()[0]],
            stem_shapes=desc.stem_shapes,
            nodes = node_descs, node_shapes=desc.node_shapes,
            post_op=cell.post_op.finalize()[0],
            out_shape=desc.out_shape,
            trainables_from = desc.trainables_from
        )
        return finalized

    def finalize_node(self, node:nn.ModuleList, node_index:int,
                      node_desc:NodeDesc, max_final_edges:int,
                      *args, **kwargs)->NodeDesc:
        # get edge ranks, if rank is None it is deemed as required
        pre_selected, edge_desc_ranks = self.get_edge_ranks(node)
        ranked_selected = self.select_edges(edge_desc_ranks, max_final_edges)
        selected_edges = pre_selected + ranked_selected
        return NodeDesc(selected_edges, node_desc.conv_params)

    def select_edges(self, edge_desc_ranks:List[Tuple[EdgeDesc, float]],
                           max_final_edges:int)->List[EdgeDesc]:
        if len(edge_desc_ranks) > max_final_edges:
            # sort by rank and pick bottom
            edge_desc_ranks.sort(key=lambda d:d[1], reverse=True)
            edge_desc_ranks = edge_desc_ranks[:max_final_edges]
        return [edr[0] for edr in edge_desc_ranks]

    def get_edge_ranks(self, node:nn.ModuleList)\
            ->Tuple[List[EdgeDesc], List[Tuple[EdgeDesc, float]]]:
        selected_edges, edge_desc_ranks = [], []
        for edge in node:
            edge_desc, rank = self.finalize_edge(edge)
            # if rank is None then it is required rank
            if rank is None:
                selected_edges.append(edge_desc) # required edge
            else: # optional edge
                edge_desc_ranks.append((edge_desc, rank))
        return selected_edges, edge_desc_ranks

    def finalize_edge(self, edge)->Tuple[EdgeDesc, Optional[float]]:
        op_desc, rank = edge._op.finalize()
        return (EdgeDesc(op_desc, edge.input_ids), rank)