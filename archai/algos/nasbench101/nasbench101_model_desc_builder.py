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

        # extract model specs from params in config
        params = conf_model_desc['params'].to_dict()
        cell_matrix = params['cell_matrix']
        vertex_ops = params['vertex_ops']
        self.num_stacks = params['num_stacks']

        self._cell_matrix, self._vertex_ops = model_matrix.prune(cell_matrix, vertex_ops)

    @overrides
    def build_cell(self, in_shapes:TensorShapesList, conf_cell:Config,
                   cell_index:int) ->CellDesc:

        stem_shapes, stems = self.build_cell_stems(in_shapes, conf_cell, cell_index)
        cell_type = self.get_cell_type(cell_index)

        if self.template is None:
            node_count = self.get_node_count(cell_index)
            in_shape = stem_shapes[0] # input shape to noded is same as cell stem
            out_shape = stem_shapes[0] # we ask nodes to keep the output shape same
            node_shapes, nodes = self.build_nodes(stem_shapes, conf_cell,
                                                  cell_index, cell_type, node_count, in_shape, out_shape)
        else:
            node_shapes, nodes = self.build_nodes_from_template(stem_shapes, conf_cell, cell_index)

        post_op_shape, post_op_desc = self.build_cell_post_op(stem_shapes,
            node_shapes, conf_cell, cell_index)

        cell_desc = CellDesc(
            id=cell_index, cell_type=self.get_cell_type(cell_index),
            conf_cell=conf_cell,
            stems=stems, stem_shapes=stem_shapes,
            nodes=nodes, node_shapes=node_shapes,
            post_op=post_op_desc, out_shape=post_op_shape,
            trainables_from=self.get_trainables_from(cell_index)
        )

        # output same shape twice to indicate s0 and s1 inputs for next cell
        in_shapes.append([post_op_shape])

        return cell_desc

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