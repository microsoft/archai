# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.nas.model_desc import ModelDesc, CellDesc, NodeDesc, OpDesc, \
                              EdgeDesc, CellType
from archai.nas.cell_builder import CellBuilder
from archai.nas.operations import MultiOp, Op
from .petridish_op import PetridishOp, TempIdentityOp


class PetridishCellBuilder(CellBuilder):
    @overrides
    def register_ops(self) -> None:
        Op.register_op('petridish_normal_op',
                    lambda op_desc, arch_params, affine:
                        PetridishOp(op_desc, arch_params, False, affine))
        Op.register_op('petridish_reduction_op',
                    lambda op_desc, arch_params, affine:
                        PetridishOp(op_desc, arch_params, True, affine))
        Op.register_op('temp_identity_op',
                    lambda op_desc, arch_params, affine:
                        TempIdentityOp(op_desc))

    @overrides
    def seed(self, model_desc:ModelDesc)->None:
        # for petridish we add one node with identity to s1
        # this will be our seed model
        for cell_desc in model_desc.cell_descs():
            node_count = len(cell_desc.nodes())
            assert node_count >= 1
            first_node = cell_desc.nodes()[0]
            # if there are no edges for 1st node, add identity to s1
            if len(first_node.edges)==0:
                op_desc = OpDesc('skip_connect',
                    params={'conv': first_node.conv_params,
                            'stride': 2 if cell_desc.cell_type == CellType.Reduction
                                        else 1},
                    in_len=1, trainables=None, children=None)
                edge = EdgeDesc(op_desc, input_ids=[1])
                first_node.edges.append(edge)

            # remove empty nodes
            new_nodes = [n.clone() for n in cell_desc.nodes()
                                   if len(n.edges)>0]
            if len(new_nodes) != len(cell_desc.nodes()):
                cell_desc.reset_nodes(new_nodes, cell_desc.node_shapes,
                                      cell_desc.post_op, cell_desc.out_shape)

            self._ensure_nonempty_nodes(cell_desc)

    def _ensure_nonempty_nodes(self, cell_desc:CellDesc):
        assert len(cell_desc.nodes()) > 0
        for node in cell_desc.nodes():
            assert len(node.edges) > 0

    @overrides
    def build(self, model_desc:ModelDesc, search_iter:int)->None:
        for cell_desc in model_desc.cell_descs():
            self._build_cell(cell_desc)

    def _build_cell(self, cell_desc:CellDesc)->None:
        reduction = (cell_desc.cell_type==CellType.Reduction)

        self._ensure_nonempty_nodes(cell_desc)

        # we operate on last node, inserting another node before it
        new_nodes = [n.clone() for n in cell_desc.nodes()]
        assert len(new_nodes) >= 1, "Petridish cell building requires at least 1 node to be present"
        petridish_node = NodeDesc(edges=[],
                                  conv_params=new_nodes[-1].conv_params)
        new_nodes.insert(len(new_nodes)-1, petridish_node)

        input_ids = list(range(len(new_nodes))) # 2 + len-2
        assert len(input_ids) >= 2
        op_desc = OpDesc('petridish_reduction_op' if reduction else 'petridish_normal_op',
                            params={
                                'conv': petridish_node.conv_params,
                                # specify strides for each input, later we will
                                # give this to each primitive
                                '_strides':[2 if reduction and j < 2 else 1 \
                                           for j in input_ids],
                            }, in_len=len(input_ids), trainables=None, children=None)
        edge = EdgeDesc(op_desc, input_ids=input_ids)
        petridish_node.edges.append(edge)

        # note that post op will be recreated which means there is no
        # warm start for post op when number of nodes changes
        cell_desc.reset_nodes(new_nodes, cell_desc.node_shapes,
                              cell_desc.post_op, cell_desc.out_shape)
