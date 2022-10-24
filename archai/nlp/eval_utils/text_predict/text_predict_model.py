# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Implements a Text Predict-based model.
"""

import functools
import os
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig

from archai_nlp.core.model import ArchaiModel
from archai.nlp.eval_utils.text_predict.text_predict_utils import LRUCache


class TextPredictModel:
    """Prepares an Archai-NLP model used for Text Predict."""

    def __init__(
        self,
        pre_trained_model_path: str,
        space_token_id: int,
        max_seq_length: Optional[int] = 30,
    ) -> None:
        """Overrides initialization method.

        Args:
            pre_trained_model_path: Path to the pre-trained model.
            space_token_id: Space token identifier.
            max_seq_length: Maximum sequence length.

        """

        self.pre_trained_model_path = pre_trained_model_path
        self.space_token_id = space_token_id
        self.max_seq_length = max_seq_length

    @functools.lru_cache(maxsize=1024)
    def _create_fixed_length_tensor(self, inputs: Tuple[int, ...]) -> torch.Tensor:
        """Creates a PyTorch-ready tensor with fixed sequence length.

        Args:
            inputs: Inputs to be converted to tensor.

        Returns:
            (torch.Tensor): Tensor with shape (batch_size x max_seq_length).

        """

        if len(inputs) == 0:
            inputs = (self.space_token_id,)
        elif len(inputs) > self.max_seq_length:
            inputs = inputs[(-1 * self.max_seq_length) :]
        elif len(inputs) < self.max_seq_length:
            inputs = (self.space_token_id,) * (self.max_seq_length - len(inputs)) + inputs

        tensor = torch.tensor(inputs).to(self.device).unsqueeze(0)

        return tensor

    def get_loss(self, input_ids: Tuple[int, ...]) -> float:
        """Calculates the model's loss.

        Args:
            input_ids: Input tokens.

        Returns:
            (float): Loss.

        """

        if len(input_ids) == 0:
            return 0.0

        with torch.no_grad():
            n_labels, loss = 0, 0.0
            for idx in range(0, len(input_ids) - 1, self.max_seq_length):
                partial_input_ids = input_ids[idx : (idx + self.max_seq_length)]
                partial_input_ids = self._create_fixed_length_tensor(partial_input_ids)

                labels = input_ids[idx + 1 : (idx + 1 + self.max_seq_length)]
                labels = self._create_fixed_length_tensor(labels)

                output = self.model(partial_input_ids, labels=labels)
                loss += torch.sum(output.loss).item()
                n_labels += len(labels)

            return loss / n_labels

    @functools.lru_cache(maxsize=1024)
    def get_next_token_probs(self, input_ids: Tuple[int, ...]) -> List[float]:
        """Calculates the probabilities of next token.

        Args:
            input_ids: Input tokens.

        Returns:
            (List[float]): Next token's probabilities.

        """

        input_ids = self._create_fixed_length_tensor(input_ids)

        with torch.no_grad():
            output = self.model(input_ids)
            next_token_probs = torch.exp(output.logits[-1][0]).tolist()

        return next_token_probs

    @functools.lru_cache(maxsize=1024)
    def get_top_next_token_probs(self, input_ids: Tuple[int, ...]) -> Tuple[int, float]:
        """Calculates the probability of top-1 next token.

        Args:
            input_ids: Input tokens.

        Returns:
            (Tuple[int, float]): Top-1 next token's identifier and probability.

        """

        probs = self.get_next_token_probs(tuple(input_ids))
        idx = np.argmax(probs)

        return (idx, probs[idx])


class TextPredictTorchModel(TextPredictModel):
    """Prepares an Archai-NLP (PyTorch) model used for Text Predict."""

    def __init__(
        self,
        pre_trained_model_path: str,
        space_token_id: int,
        max_seq_length: Optional[int] = 30,
        device: Optional[str] = None,
    ) -> None:
        """Overrides initialization method.

        Args:
            pre_trained_model_path: Path to the pre-trained model.
            space_token_id: Space token identifier.
            max_seq_length: Maximum sequence length.
            device: Device where model should be placed.

        """

        super().__init__(pre_trained_model_path, space_token_id, max_seq_length=max_seq_length)

        self.model = ArchaiModel.from_pretrained(pre_trained_model_path)
        self.device = next(self.model.parameters()).device if device is None else device

        self.model = self.model.to(self.device)
        self.model.eval()


class TextPredictONNXModel(TextPredictModel):
    """Prepares an Archai-NLP (ONNX) model used for Text Predict."""

    def __init__(
        self,
        pre_trained_model_path: str,
        space_token_id: int,
        max_seq_length: Optional[int] = 30,
    ) -> None:
        """Overrides initialization method.

        Args:
            pre_trained_model_path: Path to the pre-trained model.
            space_token_id: Space token identifier.
            max_seq_length: Maximum sequence length.

        """

        super().__init__(pre_trained_model_path, space_token_id, max_seq_length=max_seq_length)

        config_path = os.path.join(os.path.dirname(pre_trained_model_path), "config.json")
        self.config = AutoConfig.from_pretrained(config_path, local_files_only=True)

        self.sess_options = ort.SessionOptions()
        self.sess_options.intra_op_num_threads = 6
        self.sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        self.sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        self.sess_options.enable_profiling = False
        self.sess_options.log_severity_level = 4
        self.session = ort.InferenceSession(
            pre_trained_model_path, self.sess_options, providers=["CPUExecutionProvider"]
        )

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.batch_size = 1

        self.past_cache = LRUCache(maxsize=1024)
        self.min_past_cache_length = 8

    def _get_past_cache(
        self,
        input_ids: Tuple[int, ...],
        min_cutoff: Optional[int] = 1,
        max_cutoff: Optional[int] = 4,
    ) -> Tuple[List[int], int]:
        """Retrieves past key/values from cache.

        Args:
            input_ids: Input tokens.
            min_cutoff: Minimum cutoff of the cache.
            max_cutoff: Maximum cutoff of the cache.

        Returns:
            (Tuple[List[int], int]): Past key/values and their length.

        """

        if len(input_ids) - min_cutoff < self.min_past_cache_length:
            return None, len(input_ids)

        for i in range(min_cutoff, max_cutoff + 1):
            past_key = str(input_ids[: (-1 * i)])
            if past_key in self.past_cache:
                return (self.past_cache[past_key], len(input_ids[: (-1 * i)]))

        return None, len(input_ids)

    def _update_past_cache(self, input_ids: Tuple[int, ...], past_ids: List[int]) -> None:
        """Updates the past key/values cache.

        Args:
            input_ids: Input tokens.
            past_ids: Past key/values.

        """

        if len(input_ids) < self.min_past_cache_length:
            return

        self.past_cache[str(input_ids)] = past_ids

    @functools.lru_cache(maxsize=1024)
    def get_next_token_probs(self, input_ids: Tuple[int, ...]) -> List[float]:
        """Calculates the probabilities of next token.

        Args:
            input_ids: Input tokens.

        Returns:
            (List[float]): Next token's probabilities.

        """

        if len(input_ids) == 0:
            input_ids = (self.space_token_id,)
        elif len(input_ids) > self.max_seq_length:
            input_ids = input_ids[(-1 * self.max_seq_length) :]

        past_ids, past_length = self._get_past_cache(input_ids)
        if past_ids is None:
            past_sequence_length = 0
        else:
            input_ids = input_ids[past_length:]
            past_sequence_length = past_length

        ort_inputs = {}
        ort_inputs["input_ids"] = np.ascontiguousarray(
            np.array(input_ids).reshape(self.batch_size, len(input_ids))
        )

        if past_ids is None:
            past_key_values = (
                self.config.past_key_values if hasattr(self.config, "past_key_values") else 2
            )
            d_head = (
                self.config.d_head
                if hasattr(self.config, "d_head")
                else int(self.config.d_model / self.config.n_head)
            )
            past_shape = [
                past_key_values,
                self.batch_size,
                self.config.n_head,
                past_sequence_length,
                d_head,
            ]
            for i in range(self.config.n_layer):
                ort_inputs[f"past_{i}"] = np.zeros(past_shape, dtype=np.float32, order="C")
        else:
            for i in range(self.config.n_layer):
                ort_inputs[f"past_{i}"] = np.ascontiguousarray(past_ids[i])

        ort_outputs = self.session.run(None, ort_inputs)
        past_ids = ort_outputs[1:]
        probs = ort_outputs[0][0, :]

        original_input_ids = input_ids
        self._update_past_cache(original_input_ids, past_ids)

        return probs.tolist()

    def get_loss(self, input_ids: Tuple[int, ...]) -> float:
        """Calculates the model's loss.

        Args:
            input_ids: Inputs.

        Returns:
            (float): Loss.

        """

        if len(input_ids) == 0:
            return 0.0

        loss = []
        for i, token in enumerate(input_ids):
            probs = self.get_next_token_probs((self.space_token_id,) + input_ids[:i])
            loss.append(-np.log(probs[token]))

        return np.mean(loss)
