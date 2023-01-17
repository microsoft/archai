# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def gpt2_onnx_forward(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None,
) -> Dict[str, torch.FloatTensor]:
    """Forward pass through the GPT-2 model with ONNX exportability.

    This method overrides the default GPT-2 forward method and returns
    both output probabilities and past key/values.

    Args:
        input_ids: Input tensor.
        past_key_values: Past pre-computed key/values tensor.

    Returns:
        Output probabilities and past key/values.

    """

    outputs_dict = {}
    outputs = self.transformer(input_ids, past_key_values=past_key_values)

    last_hidden_state = outputs.last_hidden_state
    past_key_values = outputs.past_key_values

    logits = F.softmax(self.lm_head(last_hidden_state[:, -1, :]), dim=-1)
    outputs_dict["logits"] = logits

    if past_key_values:
        past_key_values = tuple([torch.stack(p) for p in past_key_values])
        outputs_dict["past_key_values"] = past_key_values

    return outputs_dict
