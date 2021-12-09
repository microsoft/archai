# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.

# from pathlib import Path

# import onnx
# from onnx import onnx_pb as onnx_proto
# from onnx.onnx_ml_pb2 import NodeProto
# from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
# from onnxruntime.quantization.operators.base_operator import QuantOperatorBase
# from onnxruntime.quantization.quant_utils import attribute_to_kwarg, ms_domain
# from onnxruntime.quantization.quantize import quantize_dynamic
# from onnxruntime.quantization.registry import IntegerOpsRegistry

# from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.load import create_file_name_identifier


# class GemmQuant(QuantOperatorBase):
#     """Implements a quantized version of the Gemm operator.

#     """

#     def __init__(self,
#                  onnx_quantizer: ONNXQuantizer,
#                  onnx_node: NodeProto) -> None:
#         """Overrides initialization method with custom arguments.

#         Args:
#             onnx_quantizer: An instance of the quantizer itself.
#             onnx_node: Node to be quantized.

#         """

#         super().__init__(onnx_quantizer, onnx_node)

#     def quantize(self) -> None:
#         """Quantizes a Gemm node into QGemm.

#         """

#         node = self.node
#         assert (node.op_type == 'Gemm')

#         # Updates original attributes to current node
#         kwargs = {}
#         for attribute in node.attribute:
#             kwargs.update(attribute_to_kwarg(attribute))
#         kwargs.pop('beta')

#         # Adds proper domain and missing attributes
#         kwargs['domain'] = ms_domain
#         kwargs['transA'] = 0

#         # Creates proper inputs for the QGemm node
#         (q_names, zp_names, scale_names, nodes) = self.quantizer.quantize_inputs(node,
#                                                   [0, 1],
#                                                   reduce_range=True,
#                                                   op_level_per_channel=True)

#         qgemm_inputs = []
#         for (q_name, scale_name, zp_name) in zip(q_names, scale_names, zp_names):
#             qgemm_inputs += [q_name, scale_name, zp_name]

#         # Adds a "QGemm" node to replace original Gemm with its quantized version
#         qgemm_output = node.output[0] + '_output_quantized'
#         qgemm_name = node.name + '_quant' if node.name != '' else ''
#         qgemm_node = onnx.helper.make_node('QGemm',
#                                             qgemm_inputs,
#                                             [qgemm_output],
#                                             qgemm_name,
#                                             **kwargs)
#         nodes.append(qgemm_node)

#         # Adds a "Cast" node to cast QGemm output to float
#         cast_op_output = qgemm_output + '_cast_output'
#         cast_node = onnx.helper.make_node('Cast',
#                                           [qgemm_output],
#                                           [cast_op_output],
#                                           qgemm_output + '_cast',
#                                           to=onnx_proto.TensorProto.FLOAT)
#         nodes.append(cast_node)

#         # Adds a "Add" node to sum the remaining bias to the Gemm output
#         bias_node = onnx.helper.make_node('Add',
#                                          [cast_node.output[0], 'crit.out_layers_biases.0'],
#                                          [node.output[0]],
#                                          qgemm_name + '_output_add')
#         nodes.append(bias_node)

#         # Adds new nodes to the original quantizer list
#         self.quantizer.new_nodes += nodes


# def add_new_quant_operators() -> None:
#     """Adds support for new quantization operators by changing
#     internal onnxruntime registry dictionaries.

#     """

#     # Changes the internal `IntegerOpsRegistry`
#     # and adds support for new quantization operators
#     IntegerOpsRegistry['Gemm'] = GemmQuant


# def dynamic_quantization(onnx_model_path: str) -> Path:
#     """Performs the dynamic quantization over an ONNX model.

#     Args:
#         onnx_model_path: Path to the ONNX model to be quantized.

#     Returns:
#         (Path): Path to the dynamic quantized ONNX model.

#     """

#     # Adds new quantization operators
#     # For now, we are only adding support for Gemm
#     # add_new_quant_operators()

#     # Performs the dynamic quantization
#     qnt_model_path = create_file_name_identifier(Path(onnx_model_path), '_int8')
#     quantize_dynamic(onnx_model_path,
#                      qnt_model_path,
#                      per_channel=False,
#                      reduce_range=False,
#                      optimize_model=False)

#     return qnt_model_path
