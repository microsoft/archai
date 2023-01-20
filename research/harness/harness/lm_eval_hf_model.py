# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

import torch
from harness.utils.multiple_token_stopping_criteria import MultipleTokenStoppingCriteria
from harness.utils.request_factory import Request
from lm_eval.base import BaseLM
from tqdm import tqdm
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.tokenization_utils import PreTrainedTokenizer


class HFEvalModel(BaseLM):
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, force_attention_mask: Optional[bool] = False) -> None:
        super().__init__()

        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")

        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.force_attention_mask = force_attention_mask

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        try:
            return self.model.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self.force_attention_mask:
            kwargs["attention_mask"] = torch.zeros(inps.shape, dtype=torch.long, device=inps.device)

        with torch.no_grad():
            return self.model(inps, **kwargs)[0]

    def _model_generate(self, context: str, max_length: int, eos_token_id: int) -> str:
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)

    def generate(self, requests: List[Request]) -> List[str]:
        res = []
        kwargs = {}

        for context, stop_tokens, do_sample, temperature, top_p, max_new_tokens in tqdm(requests):
            if not context:
                context = self.eos_token

            # Encodes the context and defines number of tokens to be
            # removed when generation ends (default = 1)
            input_ids = self.tokenizer(context, return_tensors="pt")["input_ids"]
            n_removal_tokens = 1

            if stop_tokens:
                # Encodes the stop-tokens and defines the number of tokens to be
                # removed as largest stop-token
                encoded_stop_tokens = self.tokenizer(
                    stop_tokens,
                    padding="longest",
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_tensors="pt",
                )["input_ids"]
                n_removal_tokens = encoded_stop_tokens.shape[-1]

                # Defines the stopping criteria
                kwargs["stopping_criteria"] = StoppingCriteriaList([MultipleTokenStoppingCriteria(encoded_stop_tokens)])

            # Generates the tokens and removes generated stop-tokens
            generated_tokens = self.model.generate(
                input_ids,
                pad_token_id=self.eot_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                **kwargs
            ).squeeze(0)
            generated_tokens = generated_tokens[:-n_removal_tokens]

            res.append(self.tok_decode(generated_tokens))

        return res
