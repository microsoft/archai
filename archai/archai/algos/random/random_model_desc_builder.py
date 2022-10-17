# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Sequence, Tuple, List, Set, Optional
import random
import copy

from overrides import overrides

from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.model_desc import ConvMacroParams, CellDesc, CellType, OpDesc, \
                                  EdgeDesc, TensorShape, TensorShapes, TensorShapesList, NodeDesc, AuxTowerDesc
from archai.common.config import Config


class RandOps:
    """Container to store (op_names, to_states) for each nodes"""
    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        # we don't allow none edge for random ops
        # 'none'  # this must be at the end so top1 doesn't choose it
    ]

    def __init__(self, n_nodes:int, max_edges:int) -> None:
        self.ops_and_ins:List[Tuple[List[str], List[int]]] = []
        for i in range(n_nodes):
            op_names = random.choices(RandOps.PRIMITIVES, k=max_edges)
            to_states = random.sample(list(range(i+2)), k=max_edges)
            self.ops_and_ins.append((op_names, to_states))


class RandomModelDescBuilder(ModelDescBuilder):
    @overrides
    def build_cells(self, in_shapes:TensorShapesList, conf_model_desc:Config)\
            ->Tuple[List[CellDesc], List[Optional[AuxTowerDesc]]]:

        max_edges = conf_model_desc['num_edges_to_sample']
        node_count = self.get_node_count(0)

        # create two sets of random ops, one for each cell type
        self._normal_ops = RandOps(node_count, max_edges)
        self._reduction_ops = RandOps(node_count, max_edges)

        return super().build_cells(in_shapes, conf_model_desc)

    @overrides
    def build_nodes(self, stem_shapes:TensorShapes, conf_cell:Config,
                    cell_index:int, cell_type:CellType, node_count:int,
                    in_shape:TensorShape, out_shape:TensorShape) \
                        ->Tuple[TensorShapes, List[NodeDesc]]:

        assert in_shape[0]==out_shape[0]

        reduction = (cell_type==CellType.Reduction)
        ops = self._reduction_ops if reduction else self._normal_ops
        assert node_count == len(ops.ops_and_ins)

        nodes:List[NodeDesc] =  []
        conv_params = ConvMacroParams(in_shape[0], out_shape[0])

        for op_names, to_states in ops.ops_and_ins:
            edges=[]
            # add random edges
            for op_name, to_state in zip(op_names, to_states):
                op_desc = OpDesc(op_name,
                                    params={
                                        'conv': conv_params,
                                        'stride': 2 if reduction and to_state < 2 else 1
                                    }, in_len=1, trainables=None, children=None)
                edge = EdgeDesc(op_desc, input_ids=[to_state])
                edges.append(edge)
            nodes.append(NodeDesc(edges=edges, conv_params=conv_params))

        out_shapes = [copy.deepcopy(out_shape) for _  in range(node_count)]

        return out_shapes, nodes






