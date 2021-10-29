# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional


class AttentionMaskFormat:
    """Enumerator for attention mask shape.

    """

    MaskIndexEnd = 0
    MaskIndexEndAndStart = 1
    AttentionMask = 2
    NoMask = 3


class FusionOptions:
    """Operators that should be fused in the ONNX graph.

    """

    def __init__(self) -> None:
        """Defines an initialization method.

        """

        # Layer normalization
        self.enable_layer_norm = True
        self.enable_skip_layer_norm = True
        self.enable_bias_skip_layer_norm = True

        # Attention
        self.enable_attention = True
        self.attention_mask_format = AttentionMaskFormat.AttentionMask

    def use_raw_attention_mask(self,
                               use_raw_mask: Optional[bool] = True) -> None:
        """Enables the usage of raw attention mask.

        Args:
            use_raw_mask: Whether raw mask should be used or not.

        """

        if use_raw_mask:
            self.attention_mask_format = AttentionMaskFormat.AttentionMask
        else:
            self.attention_mask_format = AttentionMaskFormat.MaskIndexEnd

    def disable_attention_mask(self) -> None:
        """Disable the usage of attention mask.

        """

        self.attention_mask_format = AttentionMaskFormat.NoMask
