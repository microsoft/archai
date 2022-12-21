# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX export-related utilities, such as model preparation and weight sharing."""

import types

import torch
from onnx import helper, load_model, numpy_helper, save
from onnxruntime.transformers import quantize_helper

from archai.nlp.onnx.onnx_forward import gpt2_onnx_forward


def prepare_model_for_onnx(model: torch.nn.Module, model_type: str) -> torch.nn.Module:
    """Prepare a PyTorch model for ONNX export by modifying the forward function and performing
    any additional pre-processing steps.

    Args:
        model: Instance of the model to prepare for ONNX export.
        model_type: Type of model.

    Returns:
        The prepared PyTorch model, ready for ONNX export.

    """

    # For GPT-2 architectures, we replace their forward function
    # and converts Conv1D to Linear layers
    if model_type in ["gpt2", "gpt2-flex"]:
        model.forward = types.MethodType(gpt2_onnx_forward, model)

        for layer in model.transformer.h:
            quantize_helper.conv1d_to_linear(layer.mlp)

    # Ensures evaluation model to disable dropout
    model.eval()

    return model


def weight_sharing(onnx_model_path: str, model_type: str) -> None:
    """Share weights between embedding and softmax layers in an ONNX model.

    Args:
        onnx_model_path: Path to the ONNX model that will have weights shared.
        model_type: Type of model to share the weights.

    """

    # Finds nodes in the graph based on their input name
    def _find_nodes_by_input(nodes, input_name):
        return [name for name in nodes.keys() if input_name in nodes[name].input]

    # Finds weights in the graph based on their shape
    def _find_weights_by_shape(weights, shape):
        return [name for name in weights.keys() if numpy_helper.to_array(weights[name]).shape == shape]

    # Loads the ONNX model
    model = load_model(onnx_model_path)

    # Gathers weights and nodes from the loaded model
    weights = {w.name: w for w in model.graph.initializer}
    nodes = {n.name: n for n in model.graph.node}

    if model_type in ["gpt2", "gpt2-flex"]:
        n_emb_weight = 1
        n_cutoffs = 0
    else:
        raise ValueError(f"model_type: {model_type} not supported for weight sharing.")

    for i in range(n_emb_weight):
        # Grabs the embedding weights pointer and removes from the graph
        emb_weight_name = f"word_emb.emb_layers.{i}.weight"
        if model_type in ["gpt2", "gpt2-flex"]:
            emb_weight_name = "transformer.wte.weight"

        emb_weight = numpy_helper.to_array(weights[emb_weight_name])
        model.graph.initializer.remove(weights[emb_weight_name])

        # Replaces the duplicated embedding weights by the softmax ones
        softmax_shape = (emb_weight.shape[1], emb_weight.shape[0])
        if i == 0:
            softmax_shape = (emb_weight.shape[1], emb_weight.shape[0] + n_cutoffs)
        softmax_weight = _find_weights_by_shape(weights, softmax_shape)[0]
        emb_gather_name = _find_nodes_by_input(nodes, emb_weight_name)[0]
        nodes[emb_gather_name].attribute.append(helper.make_attribute("axis", 1))
        nodes[emb_gather_name].input[0] = softmax_weight

        # Adds a "Transpose" node to invert the new embedding weights
        permute_dim = [1, 2, 0]
        if n_cutoffs != 0:
            permute_dim = [1, 0, 2]
        emb_gather_output = nodes[emb_gather_name].output[0]
        transpose_node_output = f"transposed_out_{i}"
        transpose_node = helper.make_node("Transpose", [emb_gather_output], [transpose_node_output], perm=permute_dim)
        model.graph.node.append(transpose_node)

        # Links the previous embedding output with the "Transpose" node
        emb_gather = _find_nodes_by_input(nodes, emb_gather_output)[0]
        nodes[emb_gather].input[0] = transpose_node_output

    # Saves the ONNX model
    save(model, onnx_model_path)
