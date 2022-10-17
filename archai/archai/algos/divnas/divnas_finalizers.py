# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Optional, Iterator, Dict
from overrides import overrides

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
from .divnas_cell import Divnas_Cell

class DivnasFinalizers(Finalizers):

    @overrides
    def finalize_model(self, model: Model, to_cpu=True, restore_device=True) -> ModelDesc:

        logger.pushd('finalize')

        # get config and train data loader
        # TODO: confirm this is correct in case you get silent bugs
        conf = get_conf()
        conf_loader = conf['nas']['search']['loader']
        data_loaders = get_data(conf_loader)
        assert data_loaders.train_dl is not None

        # wrap all cells in the model
        self._divnas_cells:Dict[int, Divnas_Cell] = {}
        for _, cell in enumerate(model.cells):
            divnas_cell = Divnas_Cell(cell)
            self._divnas_cells[id(cell)] = divnas_cell

        # go through all edges in the DAG and if they are of divop
        # type then set them to collect activations
        sigma = conf['nas']['search']['divnas']['sigma']
        for _, dcell in enumerate(self._divnas_cells.values()):
            dcell.collect_activations(DivOp, sigma)

        # now we need to run one evaluation epoch to collect activations
        # we do it on cpu otherwise we might run into memory issues
        # later we can redo the whole logic in pytorch itself
        # at the end of this each node in a cell will have the covariance
        # matrix of all incoming edges' ops
        model = model.cpu()
        model.eval()
        with torch.no_grad():
            for _ in range(1):
                for _, (x, _) in enumerate(data_loaders.train_dl):
                    _, _ = model(x), None
                    # now you can go through and update the
                    # node covariances in every cell
                    for dcell in self._divnas_cells.values():
                        dcell.update_covs()

        logger.popd()

        return super().finalize_model(model, to_cpu, restore_device)


    @overrides
    def finalize_cell(self, cell:Cell, cell_index:int,
                      model_desc:ModelDesc, *args, **kwargs)->CellDesc:
        # first finalize each node, we will need to recreate node desc with final version
        max_final_edges = model_desc.max_final_edges

        node_descs:List[NodeDesc] = []
        dcell = self._divnas_cells[id(cell)]
        assert len(cell.dag) == len(list(dcell.node_covs.values()))
        for i,node in enumerate(cell.dag):
            node_cov = dcell.node_covs[id(node)]
            node_desc = self.finalize_node(node, i, cell.desc.nodes()[i],
                                           max_final_edges, node_cov)
            node_descs.append(node_desc)

        # (optional) clear out all activation collection information
        dcell.clear_collect_activations()

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


    @overrides
    def finalize_node(self, node:nn.ModuleList, node_index:int,
                      node_desc:NodeDesc, max_final_edges:int,
                      *args, **kwargs)->NodeDesc:
        # node is a list of edges
        assert len(node) >= max_final_edges

        # covariance matrix shape must be square 2-D
        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1]

        # the number of primitive operators has to be greater
        # than equal to the maximum number of final edges
        # allowed
        assert cov.shape[0] >= max_final_edges

        # get total number of ops incoming to this node
        num_ops = sum([edge._op.num_valid_div_ops for edge in node])

        # and collect some bookkeeping indices
        edge_num_and_op_ind = []
        for j, edge in enumerate(node):
            if type(edge._op) == DivOp:
                for k in range(edge._op.num_valid_div_ops):
                    edge_num_and_op_ind.append((j, k))

        assert len(edge_num_and_op_ind) == num_ops

        # run brute force set selection algorithm
        max_subset, max_mi = compute_brute_force_sol(cov, max_final_edges)

        # convert the cov indices to edge descs
        selected_edges = []
        for ind in max_subset:
            edge_ind, op_ind = edge_num_and_op_ind[ind]
            op_desc = node[edge_ind]._op.get_valid_op_desc(op_ind)
            new_edge = EdgeDesc(op_desc, node[edge_ind].input_ids)
            selected_edges.append(new_edge)

        # for edge in selected_edges:
        #     self.finalize_edge(edge)

        return NodeDesc(selected_edges, node_desc.conv_params)
