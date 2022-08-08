# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ArchaiModel that works with the Text Predictor.
"""

import functools
import os
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig

from archai.common.lru_cache import LRUCache
from archai.nlp.models.model_loader import load_model_from_checkpoint


class TextPredictModel:
    """Prepares a model (PyTorch or ONNX) that complies with Text Predict.

    """

    def __init__(self,
                 model_type: str,
                 model_path: str,
                 space_token_id: int,
                 max_seq_len: int):
        self.model_type = model_type
        self.model_path = model_path

        self.space_token_id = space_token_id
        self.max_seq_len = max_seq_len

    @functools.lru_cache(maxsize=1024)
    def _ids2tensor(self, input_ids: Tuple[int, ...]) -> torch.Tensor:
        # Uses space if empty
        if len(input_ids) == 0:
            input_ids = (self.space_token_id,)

        # Uses truncation if long
        elif len(input_ids) > self.max_seq_len:
            input_ids = input_ids[(-1*self.max_seq_len):]

        # Pads if small
        n_input_ids = len(input_ids)
        if n_input_ids < self.max_seq_len:
            input_ids = (self.space_token_id,) * (self.max_seq_len - n_input_ids) + input_ids

        input_ids_t = torch.tensor(input_ids).to(self.device)
        input_ids_t = input_ids_t.unsqueeze(0)
        
        return input_ids_t

    def get_loss(self, input_ids: Tuple[int, ...]) -> float:
        if len(input_ids) == 0:
            return 0.0

        with torch.no_grad():
            n_labels = 0
            total_loss = 0.0

            for idx in range(0, len(input_ids) - 1, self.max_seq_len):
                _input_ids = input_ids[idx:(idx + self.max_seq_len)]
                labels = input_ids[idx+1:(idx + 1 + self.max_seq_len)]

                input_ids_t = self._ids2tensor(_input_ids)
                labels_t = self._ids2tensor(labels)

                loss, _, _, *_ = self.model(input_ids_t,
                                            labels=labels_t,
                                            mems=None,
                                            output_loss=True,
                                            output_prediction_scores=False)

                total_loss += torch.sum(loss).item()
                n_labels += len(labels)

            return (total_loss / n_labels)

    @functools.lru_cache(maxsize=1024)
    def get_probs(self, input_ids: Tuple[int, ...]) -> List[float]:
        input_ids_t = self._ids2tensor(input_ids)

        with torch.no_grad():
            _, prediction_scores, _, *_ = self.model(input_ids_t,
                                                     labels=None,
                                                     mems=None,
                                                     output_loss=False,
                                                     output_prediction_scores=True)

            # Takes logits for last token and gets first batch
            next_token_probs = torch.exp(prediction_scores[-1][0]).tolist()

        return next_token_probs

    @functools.lru_cache(maxsize=1024)
    def get_top_token_prob(self, input_ids: Tuple[int, ...]) -> Tuple[int, float]:
        probs = self.get_probs(tuple(input_ids))
        idx = np.argmax(probs)
        
        return (idx, probs[idx])


class TextPredictTorchModel(TextPredictModel):
    """Prepares a PyTorch model that complies with Text Predict.

    """

    def __init__(self,
                 model_type: str,
                 model_path: str,
                 space_token_id: int,
                 max_seq_len: int,
                 device: Optional[str] = None) -> None:
        super().__init__(model_type, model_path, space_token_id, max_seq_len)

        # Loads model from input checkpoint
        model, _, _ = load_model_from_checkpoint(model_type, model_path)

        # Puts model on proper device and on evaluation mode
        self.device = next(model.parameters()).device if device is None else device
        self.model = model.to(self.device)
        self.model.eval()


class TextPredictONNXModel(TextPredictModel):
    """Prepares an ONNX model that complies with Text Predict.

    """

    def __init__(self,
                 model_type: str,
                 model_path: str,
                 space_token_id: int,
                 max_seq_len: int) -> None:
        super().__init__(model_type, model_path, space_token_id, max_seq_len)

        # Loads the configuration
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        self.config = AutoConfig.from_pretrained(config_path, local_files_only=True)

        # Applies optimized ONNXRuntime session options
        self.sess_options = ort.SessionOptions()
        self.sess_options.intra_op_num_threads = 6
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.sess_options.enable_profiling = False
        self.sess_options.log_severity_level = 4

        # Creates the session
        self.session = ort.InferenceSession(model_path,
                                            self.sess_options,
                                            providers=['CPUExecutionProvider'])

        # Additional properties
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.batch_size = 1
        self.past_cache = LRUCache(maxsize=1024)
        self.min_past_cache_len = 8

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

        else:
            input_ids = input_ids[past_len:]
            past_sequence_length = past_len

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

        return probs.tolist()

    def get_past_cache(self, input_ids: tuple, min_cutoff = 1, max_cutoff = 4) -> list:
        if len(input_ids) - min_cutoff < self.min_past_cache_len:
            return None, len(input_ids)

        for i in range(min_cutoff, max_cutoff + 1):
            past_key = str(input_ids[:(-1 * i)])

            if past_key in self.past_cache:
                return (self.past_cache[past_key], len(input_ids[:(-1 * i)]))

        return None, len(input_ids)

    def update_past_cache(self, input_ids: tuple, past: list) -> None:
        if len(input_ids) < self.min_past_cache_len:
            return

        past_key = str(input_ids)
        self.past_cache[past_key] = past

    def get_loss(self, input_ids: Tuple[int, ...]) -> float:
        if len(input_ids) == 0:
            return 0.0

        next_token_logits = []
        for i, token_id in enumerate(input_ids):
            probs = self.get_probs((self.space_token_id,) + input_ids[:i])
            next_token_logits.append(-np.log(probs[token_id]))

        return np.mean(next_token_logits)
