# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.common.common import get_conf
from archai.nas.cell_builder import CellBuilder
from archai.nas.operations import Op, MultiOp
from archai.nas.model_desc import ModelDesc, CellDesc, CellType, OpDesc, EdgeDesc
from .gs_op import GsOp

class GsCellBuilder(CellBuilder):
    @overrides
    def register_ops(self) -> None:
        Op.register_op('gs_op',
                       lambda op_desc, arch_params, affine:
                           GsOp(op_desc, arch_params, affine))


    @overrides
    def build(self, model_desc:ModelDesc, search_iter:int)->None:
        # if this is not the first iteration, we add new node to each cell
        if search_iter > 0:
            self.add_node(model_desc)

        conf = get_conf()
        self._gs_num_sample = conf['nas']['search']['gs']['num_sample']

        for cell_desc in model_desc.cell_descs():
            self._build_cell(cell_desc, self._gs_num_sample)

    def _build_cell(self, cell_desc:CellDesc, gs_num_sample:int)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add gs op for each edge
        for i, node in enumerate(cell_desc.nodes()):
            for j in range(i+2):
                op_desc = OpDesc('gs_op',
                                    params={
                                        'conv': cell_desc.conv_params,
                                        'stride': 2 if reduction and j < 2 else 1,
                                        'gs_num_sample': self._gs_num_sample,
                                    }, in_len=1, trainables=None, children=None)
                edge = EdgeDesc(op_desc, input_ids=[j])
                node.edges.append(edge)



