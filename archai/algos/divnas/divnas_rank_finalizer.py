# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Optional, Iterator, Dict
from overrides import overrides

import torch
from torch import nn

import numpy as np
import seaborn as sns
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
from archai.algos.divnas.analyse_activations import compute_brute_force_sol
from archai.algos.divnas.divop import DivOp
from archai.nas.operations import Zero

from .divnas_cell import Divnas_Cell


class DivnasRankFinalizers(Finalizers):

    @overrides
    def finalize_model(self, model: Model, to_cpu=True, restore_device=True) -> ModelDesc:

        logger.pushd('finalize')

        # get config and train data loader
        conf = get_conf()
        conf_loader = conf['nas']['search']['loader']
        data_loaders = get_data(conf_loader)
        assert data_loaders.train_dl is not None

        # wrap all cells in the model
        self._divnas_cells: Dict[Cell, Divnas_Cell] = {}
        for _, cell in enumerate(model.cells):
            divnas_cell = Divnas_Cell(cell)
            self._divnas_cells[cell] = divnas_cell

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
                    # update the node covariances in all cells
                    for dcell in self._divnas_cells.values():
                        dcell.update_covs()

        logger.popd()

        return super().finalize_model(model, to_cpu, restore_device)

    @overrides
    def finalize_cell(self, cell:Cell, cell_index:int,
                      model_desc:ModelDesc, *args, **kwargs)->CellDesc:
        # first finalize each node, we will need to recreate node desc with final version
        logger.info(f'cell id {cell.desc.id}')

        max_final_edges = model_desc.max_final_edges

        node_descs: List[NodeDesc] = []
        dcell = self._divnas_cells[cell]
        assert len(cell.dag) == len(list(dcell.node_covs.values()))
        for i, node in enumerate(cell.dag):
            node_cov = dcell.node_covs[id(node)]
            logger.info(f'node {i}')
            node_desc = self.finalize_node(node, i, cell.desc.nodes()[i],max_final_edges, node_cov, cell, i)
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
                      cov:np.array, cell: Cell, node_id: int,
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

        # get the order and alpha of all ops other than 'none'
        in_ops = [(edge,op,alpha,i) for i, edge in enumerate(node) \
                            for op, alpha in edge._op.ops()
                            if not isinstance(op, Zero)]
        assert len(in_ops) >= max_final_edges

        # order all the ops by alpha
        in_ops_sorted = sorted(in_ops, key=lambda in_op:in_op[2], reverse=True)

        # keep under consideration top half of the ops
        num_to_keep = max(max_final_edges, len(in_ops_sorted)//2)
        top_ops = in_ops_sorted[:num_to_keep]

        # get the covariance submatrix of the top ops only
        cov_inds = []
        for edge, op, alpha, edge_num in top_ops:
            ind = self._divnas_cells[cell].node_num_to_node_op_to_cov_ind[node_id][op]
            cov_inds.append(ind)

        cov_top_ops = cov[np.ix_(cov_inds, cov_inds)]

        assert len(cov_inds) == len(top_ops)
        assert len(top_ops) >= max_final_edges
        assert cov_top_ops.shape[0] == cov_top_ops.shape[1]
        assert len(cov_top_ops.shape) == 2

        # run brute force set selection algorithm
        # only on the top ops
        max_subset, max_mi = compute_brute_force_sol(cov_top_ops, max_final_edges)

        # note that elements of max_subset are indices into top_ops only
        selected_edges = []
        for ind in max_subset:
            edge, op, alpha, edge_num = top_ops[ind]
            op_desc, _ = op.finalize()
            new_edge = EdgeDesc(op_desc, edge.input_ids)
            logger.info(f'selected edge: {edge_num}, op: {op_desc.name}')
            selected_edges.append(new_edge)

        # save diagnostic information to disk
        expdir = get_expdir()
        sns.heatmap(cov_top_ops, annot=True, fmt='.1g', cmap='coolwarm')
        savename = os.path.join(
            expdir, f'cell_{cell.desc.id}_node_{node_id}_cov.png')
        plt.savefig(savename)

        logger.info('')
        return NodeDesc(selected_edges, node_desc.conv_params)
