# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from collections import OrderedDict
from itertools import chain
from typing import Optional

import torch
from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM
from onnx import helper, load_model, numpy_helper, save

from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.operators import register_trilu_operator


def weight_sharing(onnx_model_path: str) -> None:
    """Shares weights between embedding and softmax layers.

    Args:
        onnx_model_path: Path to the ONNX model that will have weights shared.

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
    n_emb_weight = len(list(filter(lambda x: 'word_emb.emb_layers' in x, weights.keys())))
    n_cutoffs = n_emb_weight - 1

    for i in range(n_emb_weight):
        # Grabs the embedding weights pointer and removes from the graph
        emb_weight_name = f'word_emb.emb_layers.{i}.weight'
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


def export_onnx_from_pt(model: MemTransformerLM,
                        model_config: dict,
                        onnx_model_path: str,
                        share_weights: Optional[bool] = True,
                        do_constant_folding: Optional[bool] = True,
                        use_external_data_format: Optional[bool] = False,
                        enable_onnx_checker: Optional[bool] = True,
                        opset_version: Optional[int] = 12) -> None:
    """Exports a PyTorch-based model to ONNX.

    Args:
        model: Input model.
        model_config: Model configuration.
        onnx_model_path: Path to the output ONNX model file.
        share_weights: Whether embedding/softmax weights should be shared.
        do_constant_folding: Whether to apply constant folding.
        use_external_data_format: Whether to use external data format.
        enable_onnx_checker: Whether to enable ONNX checker.
        opset_version: Version of the operators set.

    """

    # Constant definitions
    n_layer = model_config['n_layer']
    n_token = model_config['n_token']
    n_head = model_config['n_head']
    d_head = model_config['d_head']
    attn_type = model_config['attn_type']

    # Creates the `past_key_values` mockup inputs
    if attn_type == 0:
        n_past_values = 3
    else:
        n_past_values = 2
    past_key_values = tuple([torch.zeros(n_past_values, 1, n_head, 32, d_head) for _ in range(n_layer)])

    # Defines some mockup inputs
    inputs = {
        'input_ids': torch.randint(0, n_token, (1, 32)),
        'past_key_values': past_key_values
    }

    # Defines the names of ONNX inputs and outputs
    onnx_past_inputs = [(f'past_{i}', {1: 'batch_size', 3: 'past_seq_len'}) for i in range(n_layer)]
    onnx_past_outputs = [(f'present_{i}', {1: 'batch_size', 3: 'total_seq_len'}) for i in range(n_layer)]
    onnx_inputs = OrderedDict([('input_ids', {0: 'batch_size', 1: 'seq_len'})] + onnx_past_inputs)
    onnx_outputs = OrderedDict([('probs', {0: 'batch_size'})] + onnx_past_outputs)

    # Creates the dynamic axes based on inputs and outputs
    dynamic_axes = {name: axes for name, axes in chain(onnx_inputs.items(), onnx_outputs.items())}

    # Applies a caveat to use unsupported triu/tril by PyTorch
    register_trilu_operator()

    # Exports model to ONNX
    torch.onnx.export(model,
                      (inputs,),
                      onnx_model_path,
                      input_names=list(onnx_inputs.keys()),
                      output_names=list(onnx_outputs.keys()),
                      dynamic_axes=dynamic_axes,
                      do_constant_folding=do_constant_folding,
                      use_external_data_format=use_external_data_format,
                      enable_onnx_checker=enable_onnx_checker,
                      opset_version=opset_version,
                      custom_opsets={'com.microsoft': 1})

    # Exports configuration to JSON
    model_config['model_type'] = 'transfo-xl'
    model_config['num_attention_heads'] = n_head
    model_config['past_key_values'] = n_past_values
    with open('config.json', 'w') as f:
        json.dump(model_config, f)

    # Applies weight sharing
    if share_weights:
        weight_sharing(onnx_model_path)
