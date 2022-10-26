# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-based call and factory to abstract sampling procedures.
"""

import re
from typing import Any, Dict

import torch
from transformers.generation_stopping_criteria import StoppingCriteria


class HarnessCall:
    """Implements an abstraction capable of handling any call, such as forward passes."""

    AVAILABLE_CALLS = ["cosine_similarity", "generate", "log_likelihood"]

    def __init__(self, call_name: str, args: Any, kwargs: Dict[str, Any]) -> None:
        """Initializes with custom arguments and keywords.

        Args:
            call_name: Name of the call.
            args: Arguments passed to the `call_name` method.
            kwargs: Keyword arguments passed to the `call_name` method.

        """

        assert (
            call_name in self.AVAILABLE_CALLS
        ), f"`call` should be in {self.AVAILABLE_CALLS}."

        self.call_name = call_name
        self.args = args
        self.kwargs = kwargs


class HarnessCallFactory:
    """Implements a factory capable of invoking HarnessCall instances."""

    def __getattr__(self, call_name: str) -> HarnessCall:
        """Gets an abstract HarnessCall based on supplied `call_name`.

        Args:
            call_name: Abstract call to be retrieved.

        Returns:
            (HarnessCall): An abstract call which can be invoked on-demand.

        """

        def fn(*args, **kwargs):
            return HarnessCall(call_name, args, kwargs)

        return fn


call_factory = HarnessCallFactory()


class MultipleTokenStoppingCriteria(StoppingCriteria):
    """Implements a stopping criteria capable of receiving multiple stop-tokens."""

    def __init__(self, stop_tokens: torch.LongTensor) -> None:
        """Initializes with custom arguments and keywords.

        Args:
            stop_tokens: Stop-tokens.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """Validates the last generated token.

        Args:
            input_ids: Input tokens.
            scores: Prediction scores of a language modeling head.

        Returns:
            (bool): Whether generation should stop or not.

        """

        # Only gathers the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to stop_tokens
        generated_inputs = input_ids[0, -self.max_stop_tokens :]
        equal_generated_inputs = torch.all(torch.eq(generated_inputs, self.stop_tokens), dim=1)

        return torch.any(equal_generated_inputs)


def clean_sample_text(text: str) -> str:
    """Performs pre-processing to clean lingering spaces out of the sample's text.

    Args:
        text: Text from sample.

    Returns:
        (str): Cleaned text from sample.

    """

    text = text.replace(" n't", "n't")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    text = re.sub(r" (['.,])", r"\1", text)

    return text
