# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-compliant forward functions.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def gpt2_onnx_forward(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Tuple[torch.FloatTensor, ...]] = None,
) -> Tuple[torch.FloatTensor, ...]:
    """Overrides the GPT-2 forward by returning probabilities and past key/values.

    Args:
        input_ids: Input tensor.
        past_key_values: Past pre-computed key/values tensor.

    Returns:
        (Tuple[torch.FloatTensor, ...]): Output probabilities and past key/values.

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
