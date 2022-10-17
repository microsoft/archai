# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pipeline for performing Post-Training Quantization (PTQ).
"""

from pathlib import Path
from typing import List, Optional

import onnx
import torch
from onnx import onnx_pb as onnx_proto
from onnx.onnx_ml_pb2 import NodeProto
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.operators.base_operator import QuantOperatorBase
from onnxruntime.quantization.quant_utils import attribute_to_kwarg, ms_domain
from onnxruntime.quantization.quantize import quantize_dynamic
from onnxruntime.quantization.registry import IntegerOpsRegistry

from archai.common.utils import create_file_name_identifier, rgetattr, rsetattr
from archai.nlp.models.model_loader import load_model_from_checkpoint


class GemmQuant(QuantOperatorBase):
    """Implements a quantized version of the Gemm operator.

    """

    def __init__(self,
                 onnx_quantizer: ONNXQuantizer,
                 onnx_node: NodeProto) -> None:
        """Overrides initialization method with custom arguments.

        Args:
            onnx_quantizer: An instance of the quantizer itself.
            onnx_node: Node to be quantized.

        """

        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self) -> None:
        """Quantizes a Gemm node into QGemm.

        """

        node = self.node
        assert (node.op_type == 'Gemm')

        # Updates original attributes to current node
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs.pop('beta')

        # Adds proper domain and missing attributes
        kwargs['domain'] = ms_domain
        kwargs['transA'] = 0

        # Creates proper inputs for the QGemm node
        (q_names, zp_names, scale_names, nodes) = self.quantizer.quantize_inputs(node,
                                                  [0, 1],
                                                  reduce_range=True,
                                                  op_level_per_channel=True)

        qgemm_inputs = []
        for (q_name, scale_name, zp_name) in zip(q_names, scale_names, zp_names):
            qgemm_inputs += [q_name, scale_name, zp_name]

        # Adds a "QGemm" node to replace original Gemm with its quantized version
        qgemm_output = node.output[0] + '_output_quantized'
        qgemm_name = node.name + '_quant' if node.name != '' else ''
        qgemm_node = onnx.helper.make_node('QGemm',
                                            qgemm_inputs,
                                            [qgemm_output],
                                            qgemm_name,
                                            **kwargs)
        nodes.append(qgemm_node)

        # Adds a "Cast" node to cast QGemm output to float
        cast_op_output = qgemm_output + '_cast_output'
        cast_node = onnx.helper.make_node('Cast',
                                          [qgemm_output],
                                          [cast_op_output],
                                          qgemm_output + '_cast',
                                          to=onnx_proto.TensorProto.FLOAT)
        nodes.append(cast_node)

        # Adds a "Add" node to sum the remaining bias to the Gemm output
        bias_node = onnx.helper.make_node('Add',
                                         [cast_node.output[0], 'crit.out_layers_biases.0'],
                                         [node.output[0]],
                                         qgemm_name + '_output_add')
        nodes.append(bias_node)

        # Adds new nodes to the original quantizer list
        self.quantizer.new_nodes += nodes


def add_new_quant_operators() -> None:
    """Adds support for new quantization operators by changing
    internal onnxruntime registry dictionaries.

    """

    # Changes the internal `IntegerOpsRegistry`
    # and adds support for new quantization operators
    IntegerOpsRegistry['Gemm'] = GemmQuant


def dynamic_quantization_onnx(onnx_model_path: str) -> Path:
    """Performs the dynamic quantization over an ONNX model.

    Args:
        onnx_model_path: Path to the ONNX model to be quantized.

    Returns:
        (Path): Path to the dynamic quantized ONNX model.

    """

    # Adds new quantization operators
    # For now, we are only adding support for Gemm
    # add_new_quant_operators()

    # Performs the dynamic quantization
    qnt_model_path = create_file_name_identifier(Path(onnx_model_path), '_int8')
    quantize_dynamic(onnx_model_path,
                     qnt_model_path,
                     per_channel=False,
                     reduce_range=False,
                     optimize_model=False)

    return qnt_model_path


def _dynamic_quantization_torch(model: torch.nn.Module,
                                embedding_layers: List[str]) -> None:
    """Wraps the inner core of dynamic quantization with a PyTorch model.

    Args:
        torch_model: PyTorch model to be quantized.
        embedding_layers: List with string-based identifiers of embedding layers.

    """

    # Sets the number of threads
    # Quantized model only uses maximum of 1 thread
    torch.set_num_threads(1)

    # Performs an initial dynamic quantization
    torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, inplace=True)

    # Currently, code below works as a caveat to quantize the embedding layers
    for l in embedding_layers:
        # Checks if supplied embedding layer really exists
        if rgetattr(model, l, 0):
            # Sets the appropriate `qconfig` for embedding layers
            attr = l + '.qconfig'
            rsetattr(model, attr, torch.quantization.float_qparams_weight_only_qconfig)
    
    # Prepares the model for quantization and quantizes it
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)


def dynamic_quantization_torch_from_model(model: torch.nn.Module,
                                          embedding_layers: Optional[List[str]] = ['word_emb', 'model.transformer.wpe', 'model.transformer.wte']
                                          ) -> None:
    """Performs the dynamic quantization over a PyTorch model.

    Args:
        model: PyTorch model to be quantized.
        embedding_layers: List with string-based identifiers of embedding layers.

    """

    # Performs the dynamic quantization
    _dynamic_quantization_torch(model, embedding_layers)


def dynamic_quantization_torch_from_path(model_type: str,
                                         torch_model_path: Optional[str] = None,
                                         embedding_layers: Optional[List[str]] = ['word_emb', 'model.transformer.wpe', 'model.transformer.wte']
                                         ) -> torch.nn.Module:
    """Performs the dynamic quantization over a PyTorch model path.

    Args:
        model_type: Type of model to be loaded.
        torch_model_path: Path to the PyTorch model to be quantized.
        embedding_layers: List with string-based identifiers of embedding layers.
        
    Returns:
        (torch.nn.Module): Dynamic quantized model.

    """

    # Loads the pre-trained model
    model, _, _ = load_model_from_checkpoint(model_type,
                                             torch_model_path,
                                             on_cpu=True)

    # Performs the dynamic quantization
    _dynamic_quantization_torch(model, embedding_layers)
    
    return model
