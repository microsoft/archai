# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, List
import copy

from overrides import overrides

from archai.nas.model_desc import ConvMacroParams, CellDesc, CellType, OpDesc, \
                                  EdgeDesc, TensorShape, TensorShapes, NodeDesc, ModelDesc
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.operations import MultiOp, Op
from archai.common.config import Config

from .petridish_op import PetridishOp, TempIdentityOp


class PetridishModelBuilder(ModelDescBuilder):
    @overrides
    def pre_build(self, conf_model_desc:Config)->None:
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
    def seed_cell(self, model_desc:ModelDesc)->None:
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
    def build_nodes(self, stem_shapes:TensorShapes, conf_cell:Config,
                    cell_index:int, cell_type:CellType, node_count:int,
                    in_shape:TensorShape, out_shape:TensorShape) \
                        ->Tuple[TensorShapes, List[NodeDesc]]:

        assert in_shape[0]==out_shape[0]

        reduction = (cell_type==CellType.Reduction)

        cell_template = self.get_cell_template(cell_index)
        assert cell_template is not None and len(cell_template.nodes())>0
        assert all(len(n.edges)>0 for n in cell_template.nodes())

        nodes:List[NodeDesc] = [n.clone() for n in cell_template.nodes()]

        conv_params = ConvMacroParams(in_shape[0], out_shape[0])

        input_ids = list(range(len(nodes))) # 2 + len-2
        assert len(input_ids) >= 2
        op_desc = OpDesc('petridish_reduction_op' if reduction else 'petridish_normal_op',
                            params={
                                'conv': conv_params,
                                # specify strides for each input, later we will
                                # give this to each primitive
                                '_strides':[2 if reduction and j < 2 else 1 \
                                           for j in input_ids],
                            }, in_len=len(input_ids), trainables=None, children=None)
        edge = EdgeDesc(op_desc, input_ids=input_ids)
        new_node = NodeDesc(edges=[edge], conv_params=conv_params)
        nodes.insert(len(nodes)-1, new_node)

        out_shapes = [copy.deepcopy(out_shape) for _  in range(node_count)]

        return out_shapes, nodes

