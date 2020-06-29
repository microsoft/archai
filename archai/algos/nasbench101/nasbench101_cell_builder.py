# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.nas.model_desc import ModelDesc, CellDesc, NodeDesc, OpDesc, \
                              EdgeDesc, CellType
from archai.nas.cell_builder import CellBuilder
from archai.nas.operations import MultiOp, Op

from . import model_matrix
from .nasbench101_op import NasBench101Op

class NasBench101CellBuilder(CellBuilder):
    @overrides
    def register_ops(self) -> None:
        Op.register_op('nasbench101_op',
                       lambda op_desc, arch_params, affine:
                           NasBench101Op(op_desc, arch_params, affine))

    @overrides
    def build(self, model_desc:ModelDesc, search_iter:int)->None:
        cell_matrix = model_desc.params['cell_matrix']
        vertex_ops = model_desc.params['vertex_ops']

        self._cell_matrix, self._vertex_ops = model_matrix.prune(cell_matrix, vertex_ops)

        for cell_desc in model_desc.cell_descs():
            self._build_cell(cell_desc)

    def _build_cell(self, cell_desc:CellDesc)->None:
        for i, node in enumerate(cell_desc.nodes()):
            input_ids = []
            first_proj = False # if input node is connected then it needs projection
            if self._cell_matrix[0, i+1]: # nadbench internal node starts at 1
                input_ids.append(0) # connect to s0
                first_proj = True

            for j in range(i): # look at all internal vertex before us
                if self._cell_matrix[j+1, i+1]: # if there is connection
                    input_ids.append(j+2) # offset because of s0, s1

            op_desc = OpDesc('nasbench101_op',
                                params={
                                    'conv': cell_desc.conv_params,
                                    'stride': 1,
                                    'vertex_op': self._vertex_ops[i+1], # offset because of input node
                                    'first_proj': first_proj
                                }, in_len=len(input_ids), trainables=None, children=None) # TODO: should we pass children here?
            edge = EdgeDesc(op_desc, input_ids=input_ids)
            node.edges.append(edge)
