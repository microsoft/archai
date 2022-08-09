# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""OPT-based flexible transformer.
"""

from transformers.models.opt.configuration_opt import OPTConfig as OPTCfg

from archai.nlp.core.model import ArchaiModel
from archai.nlp.models.opt.opt_flex_utils import OPTForCausalLMFlex
from archai.common.utils import map_to_list


class OPTFlexConfig(OPTCfg):
    """Wraps an OPT flexible transformer configuration."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the configuration of the transformer."""

        if "ffn_dim" in kwargs:
            kwargs["ffn_dim"] = map_to_list(kwargs["ffn_dim"], kwargs["num_hidden_layers"])

        if "num_attention_heads" in kwargs:
            kwargs["num_attention_heads"] = map_to_list(
                kwargs["num_attention_heads"], kwargs["num_hidden_layers"]
            )

        super().__init__(*args, **kwargs)


class OPTFlexForCLM(OPTForCausalLMFlex, ArchaiModel):
    """Wraps an OPT flexible transformer for causal language modeling."""

    def __init__(self, *args, **kwargs) -> None:
        """Wraps and defines the architecture of the transformer."""

        super().__init__(*args, **kwargs)
