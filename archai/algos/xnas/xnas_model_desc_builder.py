# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.operations import Op
from archai.nas.model_desc import ModelDesc, CellDesc, CellType, OpDesc, EdgeDesc
from archai.common.config import Config

from .xnas_op import XnasOp

class XnasModelDescBuilder(ModelDescBuilder):
    @overrides
    def pre_build(self, conf_model_desc:Config)->None:
        Op.register_op('xnas_op',
                       lambda op_desc, arch_params, affine:
                           XnasOp(op_desc, arch_params, affine))

    @overrides
    def build(self, model_desc:ModelDesc, search_iter:int)->None:
        # # if this is not the first iteration, we add new node to each cell
        # if search_iter > 0:
        #     self.add_node(model_desc)

        for cell_desc in model_desc.cell_descs():
            self._build_cell(cell_desc)

    def _build_cell(self, cell_desc:CellDesc)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add xnas op for each edge
        for i, node in enumerate(cell_desc.nodes()):
            for j in range(i+2):
                op_desc = OpDesc('xnas_op',
                                    params={
                                        'conv': node.conv_params,
                                        'stride': 2 if reduction and j < 2 else 1
                                    }, in_len=1, trainables=None, children=None)
                edge = EdgeDesc(op_desc, input_ids=[j])
                node.edges.append(edge)



