# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers.models.transfo_xl.configuration_transfo_xl import TransfoXLConfig


class MemTransformerConfig(TransfoXLConfig):
    model_type = "mem-transformer"

    def __init__(self, *args, **kwargs) -> None:
        if "primer_conv" not in kwargs:
            kwargs["primer_conv"] = False
        if "primer_square" not in kwargs:
            kwargs["primer_square"] = False
        if "fp16" not in kwargs:
            kwargs["fp16"] = False
        if "use_cache" not in kwargs:
            kwargs["use_cache"] = False

        super().__init__(*args, **kwargs)
