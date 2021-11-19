from typing import Tuple
import logging
import functools
import time

from scipy.special import softmax

import numpy as np

import torch
from torch import nn


class ModelWrapper:
    def __init__(self, model:nn.Module, space_token_id:int,
                 max_seq_len:int, # 64 for Micronet, 128 otherwise?
                 device=None):
        self.space_token_id = space_token_id
        self.max_seq_len = max_seq_len
        self.model = model
        self.device = next(model.parameters()).device if device is None else device
        self.model.eval()

    @functools.lru_cache(maxsize=1024)
    def _ids2tensor(self, input_ids: tuple):
        # if empty then use space
        if len(input_ids) == 0:
            input_ids = (self.space_token_id,)

        # if too long then truncate
        elif len(input_ids) > self.max_seq_len:
            input_ids = input_ids[(-1*self.max_seq_len):]

        # pad if too small
        input_ids_len = len(input_ids)
        if input_ids_len < self.max_seq_len:
            # TODO: tomasz: pad left instead of right using space IDs
            input_ids = (self.space_token_id,) * (self.max_seq_len - input_ids_len) + input_ids

        tokenized_tensor = torch.tensor(input_ids).to(self.device) # pylint: disable=not-callable
        tokenized_tensor = tokenized_tensor.unsqueeze(0) # add batch dimension
        
        return tokenized_tensor

    def get_loss(self, input_ids: tuple) -> float:
        # TODO: BUG: Few % difference from calculating manually
        # shift labels & inputs?
        if len(input_ids) == 0:
            return 0.0

        with torch.no_grad():
            labels_len_sum = 0
            loss_sum = 0.0
            for idx in range(0, len(input_ids)-1, self.max_seq_len):
                data = input_ids[idx:(idx + self.max_seq_len)]
                target = input_ids[idx+1:(idx + 1 + self.max_seq_len)]

                data_t = self._ids2tensor(data)
                target_t = self._ids2tensor(target)

                loss, mems, log_prob, *_ = self.model(data_t, target=target_t, mems=None)
                loss_sum += torch.sum(loss).item()
                labels_len_sum += len(target)
            return (loss_sum / labels_len_sum)

    @functools.lru_cache(maxsize=1024)
    def get_probs(self, input_ids: tuple) -> list:
        """Run the model with given input_ids, return probability distribution over the tokens

        Args:
            input_ids (tuple): token ids from the associated tokenizer.

        Returns:
            list: probability distribution over all tokens
        """
        start = time.time()
        in_tensor = self._ids2tensor(input_ids)

        with torch.no_grad():
            loss, mems, log_prob, *_ = self.model(in_tensor, target=None, mems=None,
                                                    return_nll=False, return_log_probs=True)
            # take logits for last token, get first batch
            next_token_probs = torch.exp(log_prob[-1][0]).tolist()

        logging.debug("Model time for %s input_ids: %s ms; first 10 probs: %s", len(input_ids), 1000*(time.time() - start), next_token_probs[:10])
        return next_token_probs

    @functools.lru_cache(maxsize=1024)
    def get_top_token_prob(self, input_ids: tuple) -> Tuple[int, float]:
        """Return id and probability of top token

        Args:
            input_ids (list): token ids from the associated tokenizer.
            This requires list for easier implementation.

        Returns:
            (int, float): idx and probability of the most likely token
        """
        probs = self.get_probs(tuple(input_ids))
        idx = np.argmax(probs)
        return (idx, probs[idx])
