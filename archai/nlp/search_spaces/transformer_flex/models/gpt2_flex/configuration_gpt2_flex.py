# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2 Flexible Transformer configuration.
"""

from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from archai.common.utils import map_to_list


class GPT2FlexConfig(GPT2Config):
    model_type = "gpt2-flex"

    def __init__(self, *args, **kwargs) -> None:
        if "n_inner" in kwargs:
            kwargs["n_inner"] = map_to_list(kwargs["n_inner"], kwargs["n_layer"])
        if "n_head" in kwargs:
            kwargs["n_head"] = map_to_list(kwargs["n_head"], kwargs["n_layer"])
        if "primer_square" not in kwargs:
            kwargs["primer_square"] = False
            
        if kwargs["primer_square"]:
            kwargs["activation_function"] = "relu"

        super().__init__(*args, **kwargs)
