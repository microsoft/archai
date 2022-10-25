# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from archai.common.common import map_to_list


class GPT2FlexConfig(GPT2Config):
    """Wraps a GPT-2 flexible transformer configuration."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the configuration of the transformer."""

        if "n_inner" in kwargs:
            kwargs["n_inner"] = map_to_list(kwargs["n_inner"], kwargs["n_layer"])

        if "n_head" in kwargs:
            kwargs["n_head"] = map_to_list(kwargs["n_head"], kwargs["n_layer"])

        super().__init__(*args, **kwargs)

        if "primer_square" not in kwargs:
            self.primer_square = False
