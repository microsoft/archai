# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-based call and factory to abstract sampling procedures."""

import re
from typing import Any, Dict

import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class HarnessCall:
    """Abstraction to handle any call, such as forward passes.

    This class implements an abstraction that allows handling any call, such as forward passes.

    """

    AVAILABLE_CALLS = ["cosine_similarity", "generate", "log_likelihood"]

    def __init__(self, call_name: str, args: Any, kwargs: Dict[str, Any]) -> None:
        """Initialize a `HarnessCall` object.

        Args:
            call_name: Name of the call.
            args: Arguments passed to the `call_name` method.
            kwargs: Keyword arguments passed to the `call_name` method.

        Raises:
            AssertionError: If `call_name` is not in `AVAILABLE_CALLS`.

        """

        assert call_name in self.AVAILABLE_CALLS, f"`call` should be in {self.AVAILABLE_CALLS}."

        self.call_name = call_name
        self.args = args
        self.kwargs = kwargs


class HarnessCallFactory:
    """Factory to invoke `HarnessCall` instances.

    This class implements a factory that allows invoking `HarnessCall` instances based
    on a supplied `call_name`.

    """

    def __getattr__(self, call_name: str) -> HarnessCall:
        """Get an abstract `HarnessCall` object based on a `call_name`.

        Args:
            call_name: Name of the abstract `HarnessCall` object to be retrieved.

        Returns:
            An abstract `HarnessCall` object that can be invoked on-demand.

        """

        def fn(*args, **kwargs):
            return HarnessCall(call_name, args, kwargs)

        return fn


call_factory = HarnessCallFactory()


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


def clean_sample_text(text: str) -> str:
    """Perform pre-processing to clean lingering spaces out of the sample's text.

    Args:
        text: The text to be cleaned.

    Returns:
        The cleaned text.

    """

    text = text.replace(" n't", "n't")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    text = re.sub(r" (['.,])", r"\1", text)

    return text
