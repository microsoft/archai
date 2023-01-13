# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizes a Text Predict evaluation tool based on text generation."""

from archai.nlp.eval.text_predict.text_predict_eval import evaluate
from archai.nlp.eval.text_predict.text_predict_model import (
    TextPredictONNXModel,
    TextPredictTorchModel,
)
from archai.nlp.eval.text_predict.text_predict_tokenizer import TextPredictTokenizer
