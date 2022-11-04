# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-based model.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.models.auto.tokenization_auto import AutoTokenizer

from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.eval.harness.harness_utils import MultipleTokenStoppingCriteria


class HarnessModel:
    """Prepares an Archai-NLP model used for harness-based evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
    ) -> None:
        """Initializes with custom arguments and keyword arguments.

        Args:
            model: Pre-trained model.
            tokenizer: Pre-trained tokenizer.

        """

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, **kwargs) -> Tuple[torch.FloatTensor, ...]:
        """Performs a forward pass over the pre-trained model
            without storing the gradients.

        Returns:
            (Tuple[torch.FloatTensor, ...]): Model outputs.

        """

        with torch.no_grad():
            return self.model(**kwargs)

    @property
    def device(self) -> str:
        """Device to compute forward passes.

        Returns:
            (str): Device.

        """

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    @property
    def eos_token(self) -> str:
        """End-of-sentence token.

        Returns:
            (str): End-of-sentence token.

        """

        return self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> int:
        """End-of-sentence token identifier.

        Returns:
            (int): End-of-sentence token identifier.

        """

        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Maximum length of model.

        Returns:
            (int): Maximum length.

        """

        try:
            return self.model.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def model_name(self) -> str:
        """Name of the model.

        Returns:
            (str): Model's name.

        """

        return self.model.config.model_name

    def encode(self, text: Union[str, List[str]], **kwargs) -> torch.LongTensor:
        """Encodes text with the tokenizer.

        Args:
            text: Text to be encoded.

        Returns:
            (torch.LongTensor): Encoded tokens.

        """

        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
            **kwargs,
        ).input_ids.to(self.device)

    def decode(self, tokens: List[int]) -> str:
        """Decodes text with the tokenizer.

        Args:
            tokens: Tokens to be decoded.

        Returns:
            (str): Decoded tokens.

        """

        return self.tokenizer.decode(tokens)

    def cosine_similarity(self, sample: str, target: str) -> float:
        """Computes the cosine similarity between pair of samples.

        Args:
            sample: First sample in the pair.
            target: Second sample in the pair.

        Returns:
            (float): Cosine similarity between `sample` and `target`.

        """

        encoded_sample, encoded_target = self.encode([sample, target], padding="longest")

        sample_states = self(input_ids=encoded_sample.unsqueeze(0), output_hidden_states=True).hidden_states[-1]
        target_states = self(input_ids=encoded_target.unsqueeze(0), output_hidden_states=True).hidden_states[-1]

        similarity = F.cosine_similarity(sample_states, target_states, dim=2)

        return float(similarity.cpu().mean())

    def generate(self, context: str, stop_tokens: Optional[List[str]] = None, **kwargs) -> str:
        """Generates a set of tokens from a context.

        Args:
            context: Context used as prompt for the generation.
            stop_tokens: Whether generation should stop at particular tokens.

        Returns:
            (str): String representing the generated tokens.

        """

        if not context:
            context = self.eos_token

        # Encodes the context and defines number of tokens to be
        # removed when generation ends (default = 1)
        input_ids = self.encode(context)
        n_removal_tokens = 1

        if stop_tokens:
            # Encodes the stop-tokens and defines the number of tokens to be
            # removed as largest stop-token
            encoded_stop_tokens = self.encode(stop_tokens, padding="longest")
            n_removal_tokens = encoded_stop_tokens.shape[-1]

            # Defines the stopping criteria
            kwargs["stopping_criteria"] = StoppingCriteriaList([MultipleTokenStoppingCriteria(encoded_stop_tokens)])

        # Generates the tokens and removes generated stop-tokens
        generated_tokens = self.model.generate(input_ids, pad_token_id=self.eos_token_id, **kwargs).squeeze(0)
        generated_tokens = generated_tokens[:-n_removal_tokens]

        return self.decode(generated_tokens)

    def log_likelihood(self, context: str, target: str) -> float:
        """Computes the log-likelihood of generating a target from context.

        Args:
            context: Context used as prompt for the generation.
            target: Target to be achieved with the generation.

        Returns:
            (float): Log-likelihood of achieving target from context.

        """

        if not context:
            context = self.eos_token

        encoded_context = self.encode(context)
        encoded_target = self.encode(target)

        # Truncates the `input_ids` from the left to keep `max_length` constant
        # Removes the last token as it will be the predicted one
        input_ids = torch.cat((encoded_context, encoded_target), dim=1)[:, -(self.max_length + 1) :]
        input_ids = input_ids[:, :-1]

        # Calculates the truncated inputs and original target lengths
        sequence_length = input_ids.shape[-1]
        target_length = encoded_target.shape[-1]

        # Performs the forward pass to retrieve the `logits`
        # and calculates their log-probabilities
        logits = self(input_ids=input_ids).logits
        probs = F.log_softmax(logits, dim=-1)

        # Slices to original sequence length (without target)
        # and retrieves log-probabilities at corresponding target indices
        probs = probs[:, sequence_length - target_length : sequence_length, :]
        probs = torch.gather(probs, 2, encoded_target.unsqueeze(-1)).squeeze(-1)

        return float(probs.cpu().sum())
