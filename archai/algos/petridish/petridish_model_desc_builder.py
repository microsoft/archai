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
from archai.algos.random.random_model_desc_builder import RandomModelDescBuilder

from .petridish_op import PetridishOp, TempIdentityOp


class PetridishModelBuilder(RandomModelDescBuilder):
    @overrides
    def pre_build(self, conf_model_desc:Config)->None:
        
        super().pre_build(conf_model_desc)

        Op.register_op('petridish_normal_op',
                    lambda op_desc, arch_params, affine:
                        PetridishOp(op_desc, arch_params, False, affine))
        Op.register_op('petridish_reduction_op',
                    lambda op_desc, arch_params, affine:
                        PetridishOp(op_desc, arch_params, True, affine))
        Op.register_op('temp_identity_op',
                    lambda op_desc, arch_params, affine:
                        TempIdentityOp(op_desc))


    # @overrides
    # def build_nodes(self, stem_shapes:TensorShapes, conf_cell:Config,
    #                 cell_index:int, cell_type:CellType, node_count:int,
    #                 in_shape:TensorShape, out_shape:TensorShape) \
    #                     ->Tuple[TensorShapes, List[NodeDesc]]:

    #     # For petridish we add one node with identity to s1.
    #     # This will be our seed model to start with.
    #     # Later in PetridishSearcher, we will add one more node in parent after each sampling.

    #     assert in_shape[0]==out_shape[0]

    #     reduction = (cell_type==CellType.Reduction)

    #     # channels for conv filters
    #     conv_params = ConvMacroParams(in_shape[0], out_shape[0])

    #     # identity op to connect S1 to the node
    #     op_desc = OpDesc('skip_connect',
    #         params={'conv': conv_params,
    #                 'stride': 2 if reduction else 1},
    #         in_len=1, trainables=None, children=None)
    #     edge = EdgeDesc(op_desc, input_ids=[1])
    #     new_node = NodeDesc(edges=[edge], conv_params=conv_params)
    #     nodes = [new_node]

    #     # each node has same out channels as in channels
    #     out_shapes = [copy.deepcopy(out_shape) for _  in nodes]

    #     return out_shapes, nodes
