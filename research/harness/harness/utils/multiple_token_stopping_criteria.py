# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class MultipleTokenStoppingCriteria(StoppingCriteria):
    """A stopping criteria class for use in text generation tasks that allows
    for multiple stop-tokens to be specified.

    """

    def __init__(self, stop_tokens: torch.LongTensor) -> None:
        """Initializes the `MultipleTokenStoppingCriteria` object with the specified stop-tokens.

        Args:
            stop_tokens: The stop-tokens to use for stopping generation.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Determines whether or not text generation should be stopped based
        on the last generated token.

        Args:
            input_ids: The input tokens that have been generated so far.
            scores: The prediction scores of a language modeling head.

        Returns:
            A boolean value indicating whether or not text generation should be stopped.

        """

        # Only gathers the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to stop_tokens
        generated_inputs = input_ids[0, -self.max_stop_tokens :]
        equal_generated_inputs = torch.all(torch.eq(generated_inputs, self.stop_tokens), dim=1)

        return torch.any(equal_generated_inputs)
