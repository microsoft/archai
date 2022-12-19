# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Text Predict-based evaluation."""

from typing import Optional

import numpy as np

from archai.nlp.eval.text_predict.text_predict_model import TextPredictModel
from archai.nlp.eval.text_predict.text_predict_prediction import TextPredictionSequence
from archai.nlp.eval.text_predict.text_predict_predictor import Predictor
from archai.nlp.eval.text_predict.text_predict_tokenizer import TextPredictTokenizer


def evaluate(
    tp_model: TextPredictModel,
    tp_tokenizer: TextPredictTokenizer,
    data_file_path: str,
    output_dir: Optional[str] = "",
    max_body_length: Optional[int] = 10000,
    min_pred_length: Optional[int] = 6,
    save_step: Optional[int] = 100000,
    current_paragraph_only: Optional[bool] = False,
    min_score: Optional[float] = 1.0,
    max_score: Optional[float] = 5.0,
    score_step: Optional[float] = 0.1,
    expected_match_rate: Optional[float] = 0.5,
) -> None:
    """Evaluate a Text Predict model and tokenizer on a given data file.

    Args:
        tp_model: The Text Predict model to use for prediction.
        tp_tokenizer: The Text Predict tokenizer to use for preprocessing.
        data_file_path: The path to the input data file.
            The file should be in either .ljson or .txt format.
        output_dir: The output directory to save the results to.
        max_body_length: The maximum length of the input text.
        min_pred_length: The minimum length of the prediction.
        save_step: The number of steps between saving results.
        current_paragraph_only: Whether to only predict information from the current paragraph.
        min_score: The minimum score to consider a prediction a match.
        max_score: The maximum score to consider a prediction a match.
        score_step: The step between the minimum and maximum scores.
        expected_match_rate: The expected rate of correct predictions.

    """

    predictor = Predictor(
        tp_model,
        tp_tokenizer,
        max_body_length=max_body_length,
        min_pred_length=min_pred_length,
    )

    # Sequence is automatically loaded from file,
    # depending on the file extension (.ljson or .txt)
    sequence = TextPredictionSequence.from_file(
        data_file_path,
        save_step=save_step,
        min_score=min_score,
        current_paragraph_only=current_paragraph_only,
        min_pred_length=min_pred_length,
    )

    # Predicts and scores the sequence
    min_scores = np.arange(min_score, max_score, score_step).tolist()
    predictor.predict(sequence)
    predictor.score(sequence, min_scores, expected_match_rate)

    # Outputs information about prediction and scoring pipelines
    sequence.save(output_dir)
