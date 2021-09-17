# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import environ

# Constants available in onnxruntime
# that enables performance optimization
environ["OMP_NUM_THREADS"] = str(1)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import argparse

import torch
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)
from archai.nlp.nvidia_transformer_xl.mem_transformer_inference import MemTransformerLM, forward_predict_memtransformer


def parse_args():
    parser = argparse.ArgumentParser(description='Compares logits between ONNX and PyTorch models.')

    parser.add_argument('--onnx_model_path',
                        type=str,
                        help='Path to the pre-trained ONNX model file.')

    parser.add_argument('--torch_model_path',
                        type=str,
                        help='Path to the pre-trained PyTorch model file.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size.')

    parser.add_argument('--sequence_length',
                        type=int,
                        default=8,
                        help='Sequence length.')

    args = parser.parse_args()

    return args

def load_checkpoint(path, cuda):
    dst = f'cuda:{torch.cuda.current_device()}' if cuda else torch.device('cpu')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint

if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Adds some properties that may impact performance
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Creates the onnxruntime session (standard)
    session = InferenceSession(args.onnx_model_path, options)
    session.disable_fallback()

    # Loads the PyTorch model
    # Ensures that the configs are passed according to your model description
    
    checkpoint = load_checkpoint(args.torch_model_path, False)
    model_config = checkpoint['model_config']
    model = MemTransformerLM(**model_config)
    model.eval()
    dst = f'cuda:{torch.cuda.current_device()}'
    model.load_state_dict(checkpoint['model_state'])

    # Tokenizes the input text into tokens
    inputs = {
        'data': torch.randint(1, 100, (args.sequence_length, args.batch_size))
    }
    input_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}

    # Performs the inference and compares the outputs
    #torch_logits = model(**inputs)[0]
    print(inputs["data"].size())
    torch_logits = forward_predict_memtransformer(model, **inputs)
    onnx_logits = session.run(None, input_onnx) #[0]

    print(torch_logits.size())
    print(onnx_logits[0].shape)
    import numpy as np
    print(np.argmax(onnx_logits[0], axis=1))
    print(torch.argmax(torch_logits, dim=1))
    #print(f'Difference between logits: {(torch_logits != onnx_logits).sum() / (torch_logits.shape[0] * torch_logits.shape[-1]) * 100}%')




