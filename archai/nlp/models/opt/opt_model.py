# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""OPT-based transformer.
"""

from transformers.models.opt.configuration_opt import OPTConfig as OPTCfg
from transformers.models.opt.modeling_opt import OPTForCausalLM

from archai.nlp.core.model import ArchaiModel


class OPTConfig(OPTCfg):
    """Wraps an OPT transformer configuration."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the configuration of the transformer."""

        super().__init__(*args, **kwargs)


class OPTForCLM(OPTForCausalLM, ArchaiModel):
    """Wraps an OPT transformer for causal language modeling."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the architecture of the transformer."""

        super().__init__(*args, **kwargs)
