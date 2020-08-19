# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Sequence, Tuple, List, Set, Optional
import copy

from overrides import overrides

from archai.nas.model_desc import ConvMacroParams, CellDesc, CellType, OpDesc, \
                                  EdgeDesc, TensorShape, TensorShapes, TensorShapesList, NodeDesc, AuxTowerDesc
from archai.common.config import Config
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.operations import MultiOp, Op

from . import model_matrix
from .nasbench101_op import NasBench101Op

class NasBench101CellBuilder(ModelDescBuilder):
    @overrides
    def pre_build(self, conf_model_desc:Config)->None:
        Op.register_op('nasbench101_op',
                       lambda op_desc, arch_params, affine:
                           NasBench101Op(op_desc, arch_params, affine))

    @overrides
    def build_cells(self, in_shapes:TensorShapesList, conf_model_desc:Config)\
            ->Tuple[List[CellDesc], List[Optional[AuxTowerDesc]]]:
        params = conf_model_desc['params'].to_dict()
        cell_matrix = params['cell_matrix']
        vertex_ops = params['vertex_ops']

        self._cell_matrix, self._vertex_ops = model_matrix.prune(cell_matrix, vertex_ops)

        return super().build_cells(in_shapes, conf_model_desc)

    @overrides
    def build_nodes(self, stem_shapes:TensorShapes, conf_cell:Config,
                    cell_index:int, cell_type:CellType, node_count:int,
                    in_shape:TensorShape, out_shape:TensorShape) \
                        ->Tuple[TensorShapes, List[NodeDesc]]:

        assert in_shape[0]==out_shape[0]

        nodes:List[NodeDesc] =  []
        conv_params = ConvMacroParams(in_shape[0], out_shape[0])

        for i in range(node_count):
            edges = []
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
                                    'conv': conv_params,
                                    'stride': 1,
                                    'vertex_op': self._vertex_ops[i+1], # offset because of input node
                                    'first_proj': first_proj
                                }, in_len=len(input_ids), trainables=None, children=None) # TODO: should we pass children here?
            edge = EdgeDesc(op_desc, input_ids=input_ids)
            edges.append(edge)
            nodes.append(NodeDesc(edges=edges, conv_params=conv_params))

        out_shapes = [copy.deepcopy(out_shape) for _  in range(node_count)]
        return out_shapes, nodes