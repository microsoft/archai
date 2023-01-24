# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class MultipleTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gathers the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to stop_tokens
        generated_inputs = input_ids[0, -self.max_stop_tokens :]
        equal_generated_inputs = torch.all(torch.eq(generated_inputs, self.stop_tokens), dim=1)

        return torch.any(equal_generated_inputs)
