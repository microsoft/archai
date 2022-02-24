# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-based model that works with the Text Predictor.
"""

import functools
import os
import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

import onnxruntime as rt
from transformers import AutoConfig

from archai.common.lru_cache import LRUCache


class TextPredictONNXModel:
    """Wraps a ONNX-based model to comply with Text Preditor.

    """

    def __init__(self,
                 onnx_model_path: str,
                 space_token_id: int,
                 max_seq_len: int,
                 device: Optional[str] = None):
        self.space_token_id = space_token_id
        self.max_seq_len = max_seq_len

        self.MIN_PAST_CACHE_INPUT_LEN = 8

        json_file_path = os.path.join(os.path.dirname(onnx_model_path), 'config.json')
        self.config = AutoConfig.from_pretrained(json_file_path, local_files_only=True)

        self.device = 'cpu'

        self.num_threads = 6 # As suggested by Yan for evals

        self.sess_options = rt.SessionOptions()
        self.sess_options.intra_op_num_threads = self.num_threads
        self.sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.sess_options.enable_profiling = False
        self.sess_options.log_severity_level = 4
        self.session = rt.InferenceSession(onnx_model_path, self.sess_options, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

        self.batch_size = 1

        self.past_cache = LRUCache(maxsize=1024)

    @functools.lru_cache(maxsize=1024)
    def _ids2tensor(self, input_ids: Tuple[int, ...]) -> torch.Tensor:
        # Uses space if empty
        if len(input_ids) == 0:
            input_ids = (self.space_token_id,)

        # Uses truncation if long
        elif len(input_ids) > self.max_seq_len:
            input_ids = input_ids[(-1*self.max_seq_len):]

        # Pads if small
        input_ids_len = len(input_ids)
        if input_ids_len < self.max_seq_len:
            input_ids = (self.space_token_id,) * (self.max_seq_len - input_ids_len) + input_ids

        tokenized_tensor = torch.tensor(input_ids).to(self.device)
        tokenized_tensor = tokenized_tensor.unsqueeze(0)
        
        return tokenized_tensor

    def get_loss(self, input_ids: Tuple[int, ...]) -> float:
        if len(input_ids) == 0:
            return 0.0

        next_token_logits = []
        for i, token_id in enumerate(input_ids):
            probs = self.get_probs((self.space_token_id,) + input_ids[:i])
            next_token_logits.append(-np.log(probs[token_id]))

        return np.mean(next_token_logits)

    @functools.lru_cache(maxsize=1024)
    def get_probs(self, input_ids: Tuple[int, ...]) -> List[float]:
        if len(input_ids) == 0:
            input_ids = (self.TOKENIZER_SPACE_ID,)
        elif len(input_ids) > self.max_seq_len:
            input_ids = input_ids[(-1*self.max_seq_len):]
        past, past_len = self.get_past_cache(input_ids)
        input_ids_orig = input_ids
        if past is None:
            past_sequence_length = 0
            sequence_length = len(input_ids)
        else:
            input_ids = input_ids[past_len:]
            past_sequence_length = past_len
            sequence_length = len(input_ids)

        total_sequence_length = past_sequence_length + sequence_length

        ort_inputs = {}
        ort_inputs['input_ids'] = np.ascontiguousarray(np.array(input_ids).reshape(self.batch_size, len(input_ids)))

        if past is None:
            past_key_values = self.config.past_key_values if hasattr(self.config, 'past_key_values') else 2
            d_head = self.config.d_head if hasattr(self.config, 'd_head') else int(self.config.d_model / self.config.n_head)
            past_shape = [past_key_values, self.batch_size, self.config.n_head, past_sequence_length, d_head]
            for i in range(self.config.n_layer):
                ort_inputs[f'past_{i}'] = np.zeros(past_shape, dtype=np.float32, order='C')
        else:
            for i in range(self.config.n_layer):
                ort_inputs[f'past_{i}'] = np.ascontiguousarray(past[i])

        ort_outputs = self.session.run(None, ort_inputs)

        past = ort_outputs[1:]
        self.update_past_cache(input_ids_orig, past)

        probs = ort_outputs[0][0, :]
        # print(probs.shape)
        # print(probs)
        # end_time = time.time()
        # print(f"get_logits: len(input_ids) = {len(input_ids)} input_ids[-3:] ={input_ids[-3:]} time_diff = {1000*(end_time - start_time):.2f}")

        return probs.tolist()

    def get_past_cache(self, input_ids: tuple, min_cutoff = 1, max_cutoff = 4) -> list:

        # return None, len(input_ids)
        # print('Get past cache')
        if len(input_ids) - min_cutoff < self.MIN_PAST_CACHE_INPUT_LEN:
            # print('fail 1')
            return None, len(input_ids)

        for i in range(min_cutoff, max_cutoff + 1):
            past_key = str(input_ids[:(-1 * i)])
            if past_key in self.past_cache:
                # print('mem!')
                return (self.past_cache[past_key], len(input_ids[:(-1 * i)]))

        # print('fail 2')
        return None, len(input_ids)

    def update_past_cache(self, input_ids: tuple, past: list) -> None:
        if len(input_ids) < self.MIN_PAST_CACHE_INPUT_LEN:
            return

        past_key = str(input_ids)
        self.past_cache[past_key] = past

    @functools.lru_cache(maxsize=1024)
    def get_logits(self, input_ids: tuple) -> list:
        probs = self.get_probs(input_ids)
        return np.log(probs).tolist()

    @functools.lru_cache(maxsize=1024)
    def get_top_token_prob(self, input_ids: Tuple[int, ...]) -> Tuple[int, float]:
        probs = self.get_probs(tuple(input_ids))
        idx = np.argmax(probs)
        
        return (idx, probs[idx])
