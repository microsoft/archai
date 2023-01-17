# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch
from lm_eval.base import BaseLM
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFEvalModel(BaseLM):
    def __init__(self, pre_trained_model_path: str, hub_tokenizer_path: str):
        super().__init__()

        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")

        self.model = AutoModelForCausalLM.from_pretrained(pre_trained_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(hub_tokenizer_path)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context: str, max_length: int, eos_token_id: int) -> str:
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)
