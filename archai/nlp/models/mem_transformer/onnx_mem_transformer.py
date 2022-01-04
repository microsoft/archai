# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""NVIDIA's Memory Transformer (Transformer-XL) for ONNX.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from archai.nlp.common.constants import BATCH_SIZE, SEQ_LEN
from archai.nlp.compression.onnx.onnx_utils.fusion_options import FusionOptions
from archai.nlp.models.config_base import OnnxConfig
from onnx import (GraphProto, ModelProto, NodeProto, TensorProto,
                  ValueInfoProto, helper)
from onnxruntime.transformers.fusion_attention import (AttentionMask,
                                                       FusionAttention)
from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
from onnxruntime.transformers.fusion_reshape import FusionReshape
from onnxruntime.transformers.fusion_shape import FusionShape
from onnxruntime.transformers.fusion_skiplayernorm import (
    FusionBiasSkipLayerNormalization, FusionSkipLayerNormalization)
from onnxruntime.transformers.fusion_utils import FusionUtils
from onnxruntime.transformers.onnx_model import OnnxModel


class MemTransformerLMOnnxConfig(OnnxConfig):
    """Provides an ONNX-export configuration for MemTransformerLM.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        super().__init__(model_config)

        # Checks the type of attention to define the `past_key_values`
        if self.config['attn_type'] == 0:
            # `k`, `v` and relative embeddings
            self.config['past_key_values'] = 3
        else:
            # `k` and `v`
            self.config['past_key_values'] = 2

        self.config['model_type'] = 'transfo-xl'

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.

        Returns:
            (Dict[str, Any]): Mockups used to export with ONNX.
        
        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN)),
            'past_key_values': tuple([torch.zeros(self.config['past_key_values'], BATCH_SIZE, self.config['n_head'], SEQ_LEN, self.config['d_head']) for _ in range(self.config['n_layer'])])
        }


class MemTransformerLMOnnxModel(OnnxModel):
    """MemTransformerLM that enables addtiional ONNX optimizations.

    """

    def __init__(self,
                 model: ModelProto) -> None:
        """Overrides initialization method.

        Args:
            model: ONNX-based model.

        """

        super().__init__(model)

        self.attention_mask = AttentionMask(self)
        self.utils = FusionUtils(self)

    def change_graph_input_type(self,
                                graph: GraphProto,
                                graph_input: ValueInfoProto,
                                new_type: Optional[int] = TensorProto.INT32
                                ) -> Tuple[NodeProto, List[NodeProto]]:
        """Changes the input type of the graph and add Cast nodes if necessary.

        Args:
            graph: Graph instance.
            graph_input: Graph inputs.
            new_type: New data type.

        Returns:
            (Tuple[NodeProto, List[NodeProto]]): Cast node to be added and
                list of Cast nodes to be removed.

        """

        assert isinstance(graph, GraphProto)
        assert isinstance(graph_input, ValueInfoProto)
        assert self.find_graph_input(graph_input.name)

        if graph_input.type.tensor_type.elem_type == int(new_type):
            return None, []

        new_cast_node = None
        nodes_to_remove = []

        input_name_to_nodes = self.input_name_to_nodes()
        if graph_input.name in input_name_to_nodes:
            nodes = input_name_to_nodes[graph_input.name]

            nodes_not_cast = [node for node in nodes if node.op_type != 'Cast']
            if nodes_not_cast:
                node_name = self.create_node_name('Cast')
                output_name = node_name + '_' + graph_input.name
                new_value_info = graph.value_info.add()
                new_value_info.CopyFrom(graph_input)
                new_value_info.name = output_name
                new_cast_node = helper.make_node('Cast', [graph_input.name], [output_name],
                                                 to=int(graph_input.type.tensor_type.elem_type),
                                                 name=node_name)
                graph.node.extend([new_cast_node])

                for node in nodes_not_cast:
                    OnnxModel.replace_node_input(node, graph_input.name, output_name)

            nodes_cast = [node for node in nodes if node.op_type == 'Cast']
            for node in nodes_cast:
                if OnnxModel.get_node_attribute(node, 'to') == int(new_type):
                    self.replace_input_of_all_nodes(node.output[0], graph_input.name)
                if not self.find_graph_output(node.output[0]):
                    nodes_to_remove.append(node)
            if nodes_to_remove:
                self.remove_nodes(nodes_to_remove)

        graph_input.type.tensor_type.elem_type = int(new_type)

        return new_cast_node, nodes_to_remove

    def change_graph_inputs_to_int32(self) -> None:
        """Changes the inputs to int32.

        """

        graph = self.graph()

        add_cast_count = 0
        remove_cast_count = 0

        for graph_input in graph.input:
            new_node, removed_nodes = self.change_graph_input_type(graph,
                                                                   graph_input,
                                                                   TensorProto.INT32)
            if new_node:
                add_cast_count += 1
            remove_cast_count += len(removed_nodes)

    def fuse_layer_norm(self) -> None:
        """Fuses the appropriate nodes into a LayerNormalization layer.

        """

        fusion = FusionLayerNormalization(self)
        fusion.apply()

    def fuse_skip_layer_norm(self) -> None:
        """Fuses the appropriate nodes into a SkipLayerNormalization layer.

        """

        fusion = FusionSkipLayerNormalization(self)
        fusion.apply()

    def fuse_add_bias_skip_layer_norm(self) -> None:
        """Fuses the appropriate nodes into a BiasSkipLayerNormalization layer.

        """

        fusion = FusionBiasSkipLayerNormalization(self)
        fusion.apply()

    def fuse_attention(self) -> None:
        """Fuses the appropriate nodes into an Attention layer.

        """

        fusion = FusionAttention(self, 0, 0, self.attention_mask)
        fusion.apply()

    def fuse_reshape(self) -> None:
        """Fuses the appropriate nodes into a Reshape layer.

        """

        fusion = FusionReshape(self)
        fusion.apply()

    def fuse_shape(self) -> None:
        """Fuses the appropriate nodes into a Shape layer.

        """

        fusion = FusionShape(self)
        fusion.apply()

    def use_dynamic_axes(self,
                         dynamic_batch_dim: Optional[str] = 'batch_size',
                         dynamic_seq_len: Optional[str] = 'seq_len') -> None:
        """Updates inputs and outputs shapes to use dynamic axes.

        Args:
            dynamic_batch_dim: Name of batch size dimension.
            dynamic_seq_len: Name of sequence length dimension.

        """

        graph_inputs = self.get_graph_inputs_from_fused_nodes(casted=True) \
                       + self.get_graph_inputs_from_fused_nodes(casted=False)

        for inp in self.model.graph.input:
            if inp.name in graph_inputs:
                dim_proto = inp.type.tensor_type.shape.dim[0]
                dim_proto.dim_param = dynamic_batch_dim

                if dynamic_seq_len is not None:
                    dim_proto = inp.type.tensor_type.shape.dim[1]
                    dim_proto.dim_param = dynamic_seq_len

        for out in self.model.graph.output:
            dim_proto = out.type.tensor_type.shape.dim[0]
            dim_proto.dim_param = dynamic_batch_dim

    def adjust_reshape_and_expand(self) -> None:
        """Cleans up unncessary reshape nodes.

        """

        nodes_to_remove = []

        for node in self.nodes():
            if node.op_type == 'Reshape':
                reshape_shape = self.get_constant_value(node.input[1])

                if reshape_shape is not None and reshape_shape.size == 0:
                    nodes_to_remove.extend([node])
                    self.replace_input_of_all_nodes(node.output[0], node.input[0])
                    continue

                reshape_path = self.match_parent_path(node,
                                                      ['Expand', 'Expand', 'Reshape', 'Slice'],
                                                      [0, 0, 0, 0],
                                                      self.output_name_to_node())

                if reshape_path is not None:
                    expand_node = reshape_path[-3]
                    expand_shape_value = self.get_constant_value(expand_node.input[1])

                    reshape_before_expand = reshape_path[-2]
                    shape_value = self.get_constant_value(reshape_before_expand.input[1])

                    slice_node = reshape_path[-1]

                    if expand_shape_value is not None and shape_value is not None and len(
                            expand_shape_value) == 2 and len(
                                shape_value) == 1 and expand_shape_value[1] == shape_value[0]:
                        node.input[0] = slice_node.output[0]

        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)

    def clean_graph(self) -> None:
        """Cleans the graph after fusing nodes.

        """

        output_name_to_node = self.output_name_to_node()
        nodes_to_remove = []

        for node in self.nodes():
            op_input_id = {'EmbedLayerNormalization': 1, 'ReduceSum': 0, 'Attention': 3}

            if node.op_type in op_input_id:
                i = op_input_id[node.op_type]
                parent_nodes = self.match_parent_path(node,
                                                      ['Cast', 'ConstantOfShape', 'Concat', 'Unsqueeze', 'Gather', 'Shape'],
                                                      [i, 0, 0, 0, 0, 0],
                                                      output_name_to_node)

                if parent_nodes is not None:
                    cast, constantOfShape, concat, unsqueeze, gather, shape = parent_nodes

                    if shape.input[0] == self.graph().input[0].name:
                        constantOfShape.input[0] = shape.output[0]
                        output_name_to_node = self.output_name_to_node()

            if node.op_type == 'Attention':
                parent_nodes = self.match_parent_path(node,
                                                      ['ReduceSum', 'Cast', 'ConstantOfShape', 'Shape'],
                                                      [3, 0, 0, 0],
                                                      output_name_to_node)

                if parent_nodes is not None:
                    if parent_nodes[-1].input[0] == self.graph().input[0].name:
                        attention_node = helper.make_node('Attention',
                                                          inputs=node.input[0:len(node.input) - 1],
                                                          outputs=node.output,
                                                          name=node.name + '_remove_mask')
                        attention_node.domain = 'com.microsoft'
                        attention_node.attribute.extend([helper.make_attribute('num_heads', self.num_heads)])

                        self.add_node(attention_node, self.get_graph_by_node(attention_node).name)
                        nodes_to_remove.append(node)

        self.remove_nodes(nodes_to_remove)

    def optimize(self,
                 options: Optional[FusionOptions] = None,
                 add_dynamic_axes: Optional[bool] = False) -> None:
        """Performs the additional transformer-based optimization.

        Args:
            options: Options holding which operators should be fused.
            add_dynamic_axes: Whether dynamic axes should be added.

        """

        # Fuses appropriate nodes into LayerNormalization
        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()

        # Pre-processing step
        self.adjust_reshape_and_expand()

        # Fuses appropriate nodes into Reshape
        self.fuse_reshape()

        # Fuses appropriate nodes into SkipLayerNormalization
        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()

        # Fuses appropriate nodes into Attention
        # if (options is None) or options.enable_attention:
        #     if options is not None:
        #         self.attention_mask.set_mask_format(options.attention_mask_format)
        #     self.fuse_attention()

        # Fuses appropriate nodes into Shape
        self.fuse_shape()

        # Removes useless Reshape nodes that are staling through the graph
        self.utils.remove_useless_reshape_nodes()

        # Post-processing step
        self.clean_graph()
        self.prune_graph()

        # Fuses appropriate nodes into BiasSkipLayerNormalization
        if (options is None) or options.enable_bias_skip_layer_norm:
            self.fuse_add_bias_skip_layer_norm()

        # Removes unused constants that are staling through the graph
        self.remove_unused_constant()

        # Whether dynamic axes should be used
        if add_dynamic_axes:
            self.use_dynamic_axes()
