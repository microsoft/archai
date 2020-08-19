# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Sequence, Tuple, List, Set, Optional
import copy

from overrides import overrides

from archai.common.common import get_conf
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.operations import Op, MultiOp
from archai.nas.model_desc import ConvMacroParams, CellDesc, CellType, OpDesc, \
                                  EdgeDesc, TensorShape, TensorShapes, NodeDesc, ModelDesc
from archai.common.config import Config
from .gs_op import GsOp

class GsModelDescBuilder(ModelDescBuilder):
    @overrides
    def pre_build(self, conf_model_desc:Config)->None:
        Op.register_op('gs_op',
                       lambda op_desc, arch_params, affine:
                           GsOp(op_desc, arch_params, affine))

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

        gs_num_sample = conf_cell['gs']['num_sample']

        # add gs op for each edge
        for i in range(node_count):
            edges = []
            for j in range(i+2):
                op_desc = OpDesc('gs_op',
                                    params={
                                        'conv': conv_params,
                                        'stride': 2 if reduction and j < 2 else 1,
                                        'gs_num_sample': gs_num_sample,
                                    }, in_len=1, trainables=None, children=None)
                edge = EdgeDesc(op_desc, input_ids=[j])
                edges.append(edge)
            nodes.append(NodeDesc(edges=edges, conv_params=conv_params))

        out_shapes = [copy.deepcopy(out_shape) for _  in range(node_count)]
        return out_shapes, nodes

