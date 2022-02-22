# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Archai-based models that works with the Text Predictor.
"""

import functools
import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import torch


class ModelWrapper:
    """Wraps an ArchaiModel to comply with Text Preditor.

    """

    def __init__(self,
                 model: torch.nn.Module,
                 space_token_id: int,
                 max_seq_len: int,
                 device: Optional[str] = None):
        self.space_token_id = space_token_id
        self.max_seq_len = max_seq_len

        self.device = next(model.parameters()).device if device is None else device

        print(self.device)
        
        self.model = model
        self.model.eval()

    @functools.lru_cache(maxsize=1024)
    def _ids2tensor(self, input_ids: Tuple[int, ...]) -> torch.Tensor:
        print(f'_ids2tensor : {input_ids}')
        print(self.space_token_id)
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

        print(f'_ids2tensor : {input_ids}')

        tokenized_tensor = torch.tensor(input_ids).to(self.device)
        tokenized_tensor = tokenized_tensor.unsqueeze(0)
        
        return tokenized_tensor

    def get_loss(self, input_ids: Tuple[int, ...]) -> float:
        if len(input_ids) == 0:
            return 0.0

        with torch.no_grad():
            labels_len_sum = 0
            loss_sum = 0.0

            for idx in range(0, len(input_ids) - 1, self.max_seq_len):
                _input_ids = input_ids[idx:(idx + self.max_seq_len)]
                labels = input_ids[idx+1:(idx + 1 + self.max_seq_len)]

                input_ids_t = self._ids2tensor(_input_ids)
                labels_t = self._ids2tensor(labels)

                loss, _, _, *_ = self.model(input_ids_t, labels=labels_t, mems=None)
                loss_sum += torch.sum(loss).item()
                labels_len_sum += len(labels)

            return (loss_sum / labels_len_sum)

    @functools.lru_cache(maxsize=1024)
    def get_probs(self, input_ids: Tuple[int, ...]) -> List[float]:
        start = time.time()
        print(f'get_probs: {input_ids}')
        in_tensor = self._ids2tensor(input_ids)

        with torch.no_grad():
            _, prediction_scores, _, *_ = self.model(in_tensor,
                                                     labels=None,
                                                     mems=None,
                                                     output_loss=False,
                                                     output_prediction_scores=True)

            # Takes logits for last token and gets first batch
            next_token_probs = torch.exp(prediction_scores[-1][0]).tolist()

        logging.debug('Model time for %s input_ids: %s ms; first 10 probs: %s', len(input_ids), 1000*(time.time() - start), next_token_probs[:10])

        return next_token_probs

    @functools.lru_cache(maxsize=1024)
    def get_top_token_prob(self, input_ids: Tuple[int, ...]) -> Tuple[int, float]:
        probs = self.get_probs(tuple(input_ids))
        idx = np.argmax(probs)
        
        return (idx, probs[idx])
