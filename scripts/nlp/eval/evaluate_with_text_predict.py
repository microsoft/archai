# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os

from transformers import AutoModelForCausalLM

from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.eval.text_predict import (
    TextPredictONNXModel,
    TextPredictTokenizer,
    TextPredictTorchModel,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates pre-trained models using eval/text_predict package.")

    parser.add_argument("pre_trained_model_path", type=str, help="Path to the pre-trained model file.")

    parser.add_argument("data_file_path", type=str, help="Path to the data file to be predicted.")

    parser.add_argument(
        "hub_tokenizer_path",
        type=str,
        help="Name or path to the Hub's tokenizer.",
    )

    parser.add_argument(
        "-msl",
        "--max_seq_length",
        type=int,
        default=30,
        help="Maximum length of sequence.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Defines an output folder for the saved outputs.",
    )

    parser.add_argument(
        "-mbl",
        "--max_body_length",
        type=int,
        default=10000,
        help="Maximum length of the input text..",
    )

    parser.add_argument(
        "-mpl",
        "--min_pred_length",
        type=int,
        default=6,
        help="Minimum length of the prediction..",
    )

    parser.add_argument(
        "-ss",
        "--save_step",
        type=int,
        default=100000,
        help="Amount of steps to save results..",
    )

    parser.add_argument(
        "-cpo",
        "--current_paragraph_only",
        action="store_true",
        help="Only predicts information from current paragraph.",
    )

    parser.add_argument(
        "-mis",
        "--min_score",
        type=float,
        default=1.0,
        help="Minimum score.",
    )

    parser.add_argument(
        "-mas",
        "--max_score",
        type=float,
        default=5.0,
        help="Maximum score.",
    )

    parser.add_argument(
        "-scs",
        "--score_step",
        type=float,
        default=0.1,
        help="Step between minimum and maximum scores..",
    )

    parser.add_argument(
        "-emr",
        "--expected_match_rate",
        type=float,
        default=0.5,
        help="Expected match rate..",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Loads pre-trained tokenizer and wraps for Text Predict
    tokenizer = ArchaiPreTrainedTokenizerFast.from_pretrained(args.hub_tokenizer_path)
    tp_tokenizer = TextPredictTokenizer(tokenizer)

    # Loads wrapped model (ONNX or PyTorch) according to file extension
    space_token_id = tp_tokenizer.encode(" ")[0]
    if os.path.splitext(args.pre_trained_model_path)[1] == ".onnx":
        tp_model = TextPredictONNXModel(args.pre_trained_model_path, space_token_id, max_seq_length=args.max_seq_length)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.pre_trained_model_path)
        tp_model = TextPredictTorchModel(model, space_token_id, max_seq_length=args.max_seq_length)

    # Runs the Text Predict pipeline
    evaluate(
        tp_model,
        tp_tokenizer,
        args.data_file_path,
        output_dir=args.output_dir,
        max_body_length=args.max_body_length,
        min_pred_length=args.min_pred_length,
        save_step=args.save_step,
        current_paragraph_only=args.current_paragraph_only,
        min_score=args.min_score,
        max_score=args.max_score,
        score_step=args.score_step,
        expected_match_rate=args.expected_match_rate,
    )
