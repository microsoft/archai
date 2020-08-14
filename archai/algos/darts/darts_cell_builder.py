# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.nas.cell_builder import CellBuilder
from archai.nas.operations import Op
from archai.nas.model_desc import ModelDesc, CellDesc, CellType, OpDesc, EdgeDesc
from archai.algos.darts.mixed_op import MixedOp

class DartsCellBuilder(CellBuilder):
    @overrides
    def register_ops(self) -> None:
        Op.register_op('mixed_op',
                       lambda op_desc, arch_params, affine:
                           MixedOp(op_desc, arch_params, affine))

    @overrides
    def build(self, model_desc:ModelDesc, search_iter:int)->None:
        for cell_desc in model_desc.cell_descs():
            self._build_cell(cell_desc)

    def _build_cell(self, cell_desc:CellDesc)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        # add mixed op for each edge in each node
        # how does the stride works? For all ops connected to s0 and s1, we apply
        # reduction in WxH. All ops connected elsewhere automatically gets
        # reduced WxH (because all subsequent states are derived from s0 and s1).
        # Note that channel is increased via conv_params for the cell
        for i, node in enumerate(cell_desc.nodes()):
            for j in range(i+2):
                op_desc = OpDesc('mixed_op',
                                    params={
                                        'conv': node.conv_params,
                                        'stride': 2 if reduction and j < 2 else 1
                                    }, in_len=1, trainables=None, children=None)
                edge = EdgeDesc(op_desc, input_ids=[j])
                node.edges.append(edge)



