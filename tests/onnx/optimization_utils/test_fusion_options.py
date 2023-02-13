# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.onnx.optimization_utils.fusion_options import (
    AttentionMaskFormat,
    FusionOptions,
)


def test_attention_mask_format():
    # Assert that the values of the enum are as expected
    assert AttentionMaskFormat.MaskIndexEnd == 0
    assert AttentionMaskFormat.MaskIndexEndAndStart == 1
    assert AttentionMaskFormat.AttentionMask == 2
    assert AttentionMaskFormat.NoMask == 3


def test_fusion_options():
    # Assert that the default values of the options are as expected
    fusion_options = FusionOptions("some_model_type")
    assert fusion_options.enable_shape_inference is True
    assert fusion_options.enable_qordered_matmul is True
    assert fusion_options.enable_gelu is True
    assert fusion_options.enable_bias_gelu is True
    assert fusion_options.enable_gelu_approximation is False
    assert fusion_options.enable_gemm_fast_gelu is False
    assert fusion_options.enable_layer_norm is True
    assert fusion_options.enable_embed_layer_norm is True
    assert fusion_options.enable_skip_layer_norm is True
    assert fusion_options.enable_bias_skip_layer_norm is True
    assert fusion_options.enable_attention is True
    assert fusion_options.use_multi_head_attention is False
    assert fusion_options.attention_mask_format == AttentionMaskFormat.AttentionMask


def test_fusion_options_gpt2_model_type():
    # Assert that the default values of the options are as expected for `gpt2` model type
    fusion_options = FusionOptions("gpt2")
    assert fusion_options.enable_embed_layer_norm is False
    assert fusion_options.enable_skip_layer_norm is False


def test_fusion_options_use_raw_attention_mask():
    fusion_options = FusionOptions("some_model_type")

    # Assert that the default value of the option is as expected
    fusion_options.use_raw_attention_mask()
    assert fusion_options.attention_mask_format == AttentionMaskFormat.AttentionMask

    # Assert that the value of the option is as expected
    fusion_options.use_raw_attention_mask(False)
    assert fusion_options.attention_mask_format == AttentionMaskFormat.MaskIndexEnd


def test_fusion_options_disable_attention_mask():
    fusion_options = FusionOptions("some_model_type")

    # Assert that the default value of the option is as expected
    fusion_options.disable_attention_mask()
    assert fusion_options.attention_mask_format == AttentionMaskFormat.NoMask
