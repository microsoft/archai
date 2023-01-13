# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Text Predict-based model."""

import functools
import os
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig

from archai.nlp.eval.text_predict.text_predict_utils import LRUCache


class TextPredictModel:
    """Wrapper for a model used in the Text Predict framework."""

    def __init__(
        self,
        space_token_id: int,
        max_seq_length: Optional[int] = 30,
    ) -> None:
        """Initialize the `TextPredictModel` with the given arguments.

        Args:
            space_token_id: The identifier for the space token.
            max_seq_length: The maximum length of the input sequence.

        """

        self.space_token_id = space_token_id
        self.max_seq_length = max_seq_length

    @functools.lru_cache(maxsize=1024)
    def _create_fixed_length_tensor(self, inputs: Tuple[int, ...]) -> torch.Tensor:
        """Create a PyTorch-compatible tensor with a fixed sequence length.

        Args:
            inputs: The input tokens to be converted to a tensor.

        Returns:
            A tensor with shape (batch_size x max_seq_length).

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
        """Calculate the loss of the model on the given input tokens.

        Args:
            input_ids: The input tokens.

        Returns:
            The loss.

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
        """Calculate the probabilities of next token.

        Args:
            input_ids: The input tokens.

        Returns:
            Next token's probabilities.

        """

        input_ids = self._create_fixed_length_tensor(input_ids)

        with torch.no_grad():
            output = self.model(input_ids)
            next_token_probs = torch.exp(output.logits[-1][0]).tolist()

        return next_token_probs

    @functools.lru_cache(maxsize=1024)
    def get_top_next_token_probs(self, input_ids: Tuple[int, ...]) -> Tuple[int, float]:
        """Calculate the probability of top-1 next token.

        Args:
            input_ids: The input tokens.

        Returns:
            Tuple with top-1 next token and probability.

        """

        probs = self.get_next_token_probs(tuple(input_ids))
        idx = np.argmax(probs)

        return (idx, probs[idx])


class TextPredictTorchModel(TextPredictModel):
    """Wrapper for a PyTorch model used in the Text Predict framework."""

    def __init__(
        self,
        model: torch.nn.Module,
        space_token_id: int,
        max_seq_length: Optional[int] = 30,
        device: Optional[str] = None,
    ) -> None:
        """Override initialization method.

        Args:
            model: A PyTorch model.
            space_token_id: The space token identifier.
            max_seq_length: The maximum sequence length.
            device: The device where the model should be placed.

        """

        super().__init__(space_token_id, max_seq_length=max_seq_length)

        self.model = model
        self.device = next(self.model.parameters()).device if device is None else device

        self.model = self.model.to(self.device)
        self.model.eval()


class TextPredictONNXModel(TextPredictModel):
    """Wrapper for an ONNX model used in the Text Predict framework."""

    def __init__(
        self,
        onnx_model_path: str,
        space_token_id: int,
        max_seq_length: Optional[int] = 30,
    ) -> None:
        """Override initialization method.

        Args:
            onnx_model_path: A path to the ONNX model file.
            space_token_id: The space token identifier.
            max_seq_length: The maximum sequence length.

        """

        super().__init__(space_token_id, max_seq_length=max_seq_length)

        config_path = os.path.join(os.path.dirname(onnx_model_path), "config.json")
        self.config = AutoConfig.from_pretrained(config_path, local_files_only=True)

        self.sess_options = ort.SessionOptions()
        self.sess_options.intra_op_num_threads = 6
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.sess_options.enable_profiling = False
        self.sess_options.log_severity_level = 4
        self.session = ort.InferenceSession(onnx_model_path, self.sess_options, providers=["CPUExecutionProvider"])

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
        """Retrieve past key/values from cache.

        Args:
            input_ids: The input tokens.
            min_cutoff: The minimum cutoff of the cache.
            max_cutoff: The maximum cutoff of the cache.

        Returns:
            Tuple with past key/values and their length.

        """

        if len(input_ids) - min_cutoff < self.min_past_cache_length:
            return None, len(input_ids)

        for i in range(min_cutoff, max_cutoff + 1):
            past_key = str(input_ids[: (-1 * i)])
            if past_key in self.past_cache:
                return (self.past_cache[past_key], len(input_ids[: (-1 * i)]))

        return None, len(input_ids)

    def _update_past_cache(self, input_ids: Tuple[int, ...], past_ids: List[int]) -> None:
        """Update the past key/values cache.

        Args:
            input_ids: The input tokens.
            past_ids: The past key/values.

        """

        if len(input_ids) < self.min_past_cache_length:
            return

        self.past_cache[str(input_ids)] = past_ids

    @functools.lru_cache(maxsize=1024)
    def get_next_token_probs(self, input_ids: Tuple[int, ...]) -> List[float]:
        """Calculate the probabilities of next token.

        Args:
            input_ids: The input tokens.

        Returns:
            A list with the next tokens' probabilities.

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
        ort_inputs["input_ids"] = np.ascontiguousarray(np.array(input_ids).reshape(self.batch_size, len(input_ids)))

        if past_ids is None:
            past_key_values = self.config.past_key_values if hasattr(self.config, "past_key_values") else 2
            d_head = (
                self.config.d_head if hasattr(self.config, "d_head") else int(self.config.d_model / self.config.n_head)
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
        """Calculate the model's loss.

        Args:
            input_ids: The input tokens.

        Returns:
            The loss.

        """

        if len(input_ids) == 0:
            return 0.0

        loss = []
        for i, token in enumerate(input_ids):
            probs = self.get_next_token_probs((self.space_token_id,) + input_ids[:i])
            loss.append(-np.log(probs[token]))

        return np.mean(loss)
