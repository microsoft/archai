# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2 Flexible Transformer configuration."""

from typing import Optional

from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from archai.common.utils import map_to_list


class GPT2FlexConfig(GPT2Config):
    model_type = "gpt2-flex"

    def __init__(self, *args, primer_square: Optional[bool] = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.primer_square = primer_square
        if primer_square:
            self.activation_function = "relu"

        self.n_inner = self.n_inner if self.n_inner is not None else 4 * self.n_embd
        self.n_inner = map_to_list(self.n_inner, self.n_layer)

        self.n_head = map_to_list(self.n_head, self.n_layer)
