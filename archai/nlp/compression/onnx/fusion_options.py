# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-related options to fuse nodes.
"""

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

    def __init__(self, model_type: str) -> None:
        """Defines an initialization method.

        Args:
            model_type: Type of model to be fused.

        """

        # GeLU
        self.enable_gelu = True
        self.enable_bias_gelu = True
        self.enable_gelu_approximation = False

        # Layer normalization
        self.enable_layer_norm = True
        self.enable_embed_layer_norm = True
        self.enable_skip_layer_norm = True
        self.enable_bias_skip_layer_norm = True

        # Attention
        self.enable_attention = True
        self.attention_mask_format = AttentionMaskFormat.AttentionMask

        if model_type == 'hf_gpt2':
            self.enable_embed_layer_norm = False
            self.enable_skip_layer_norm = False

    def use_raw_attention_mask(self, use_raw_mask: Optional[bool] = True) -> None:
        """Enables the usage of raw attention mask.

        Args:
            use_raw_mask: Whether raw mask should be used or not.

        """

        if use_raw_mask:
            self.attention_mask_format = AttentionMaskFormat.AttentionMask
        else:
            self.attention_mask_format = AttentionMaskFormat.MaskIndexEnd

    def disable_attention_mask(self) -> None:
        """Disables the usage of attention mask.

        """

        self.attention_mask_format = AttentionMaskFormat.NoMask
