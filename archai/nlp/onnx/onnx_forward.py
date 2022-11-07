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

    outputs = self.transformer(input_ids, past_key_values=past_key_values)
    hidden_states = outputs[0]
    preds = F.softmax(self.lm_head(hidden_states[:, -1, :]), dim=-1)

    if outputs.past_key_values:
        return {
            "logits": preds,
            "past_key_values": tuple([torch.stack(p) for p in outputs.past_key_values])
        }
    
    return {
        "logits": preds
    }
