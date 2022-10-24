# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Handles every ONNX-related export methods.
"""

import json
from itertools import chain
from pathlib import Path
from typing import Optional

import torch
from onnx import helper, load_model, numpy_helper, save

from archai.nlp.models.model_loader import load_onnx_config
from archai.nlp.compression.onnx.onnx_utils.operators import (tril_onnx,
                                                              triu_onnx)


def weight_sharing(onnx_model_path: str, model_type: str) -> None:
    """Shares weights between embedding and softmax layers.

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
    weights = {w.name:w for w in model.graph.initializer}
    nodes = {n.name:n for n in model.graph.node}

    if model_type in ['hf_gpt2', 'hf_gpt2_flex']:
        n_emb_weight = 1
        n_cutoffs = 0
    elif model_type == 'mem_transformer':
        n_emb_weight = len(list(filter(lambda x: 'word_emb.emb_layers' in x, weights.keys())))
        n_cutoffs = n_emb_weight - 1
    else:
        raise ValueError(f'model_type: {model_type} not supported for weight sharing.')

    for i in range(n_emb_weight):
        # Grabs the embedding weights pointer and removes from the graph
        emb_weight_name = f'word_emb.emb_layers.{i}.weight'
        if model_type in ['hf_gpt2', 'hf_gpt2_flex']:
            emb_weight_name = 'transformer.wte.weight'

        emb_weight = numpy_helper.to_array(weights[emb_weight_name])
        model.graph.initializer.remove(weights[emb_weight_name])

        # Replaces the duplicated embedding weights by the softmax ones
        softmax_shape = (emb_weight.shape[1], emb_weight.shape[0])
        if i == 0:
            softmax_shape = (emb_weight.shape[1], emb_weight.shape[0] + n_cutoffs)
        softmax_weight = _find_weights_by_shape(weights, softmax_shape)[0]
        emb_gather_name = _find_nodes_by_input(nodes, emb_weight_name)[0]
        nodes[emb_gather_name].attribute.append(helper.make_attribute('axis', 1))
        nodes[emb_gather_name].input[0] = softmax_weight

        # Adds a "Transpose" node to invert the new embedding weights
        permute_dim = [1, 2, 0]
        if n_cutoffs != 0:
            permute_dim = [1, 0, 2]
        emb_gather_output = nodes[emb_gather_name].output[0]
        transpose_node_output = f'transposed_out_{i}'
        transpose_node = helper.make_node('Transpose', [emb_gather_output], [transpose_node_output], perm=permute_dim)
        model.graph.node.append(transpose_node)

        # Links the previous embedding output with the "Transpose" node
        emb_gather = _find_nodes_by_input(nodes, emb_gather_output)[0]
        nodes[emb_gather].input[0] = transpose_node_output

    # Saves the ONNX model
    save(model, onnx_model_path)


def export_onnx_from_torch(model: torch.nn.Module,
                           model_config: dict,
                           model_type: str,
                           onnx_model_path: str,
                           share_weights: Optional[bool] = True,
                           do_constant_folding: Optional[bool] = True,
                           opset_version: Optional[int] = 11) -> None:
    """Exports a PyTorch-based model to ONNX.

    Args:
        model: Input model.
        model_config: Model configuration.
        model_type: Type of model to be exported.
        onnx_model_path: Path to the output ONNX model file.
        share_weights: Whether embedding/softmax weights should be shared.
        do_constant_folding: Whether to apply constant folding.
        opset_version: Version of the operators set.

    """

    # Gathers the proper ONNX configuration instance
    onnx_config = load_onnx_config(model_type, model_config)

    # Creates the dynamic axes based on inputs and outputs
    dynamic_axes = {name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())}

    # Applies a caveat to use unsupported triu/tril by PyTorch
    torch.triu = triu_onnx
    torch.tril = tril_onnx

    # Exports model to ONNX
    torch.onnx.export(model,
                      (onnx_config.mockups,),
                      onnx_model_path,
                      input_names=list(onnx_config.inputs.keys()),
                      output_names=list(onnx_config.outputs.keys()),
                      dynamic_axes=dynamic_axes,
                      do_constant_folding=do_constant_folding,
                      opset_version=opset_version)

    # Exports configuration to JSON
    config_path = Path(onnx_model_path).parent / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(onnx_config.config.to_dict(), f)

    # Applies weight sharing
    if share_weights:
        weight_sharing(onnx_model_path, model_type)
