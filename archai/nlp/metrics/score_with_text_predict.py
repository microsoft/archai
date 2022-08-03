# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Score models using Text Predict.
"""

import argparse
import os

from archai.nlp.metrics.text_predict.predictor import run_score


def _check_amlt_paths(args: argparse.Namespace) -> argparse.Namespace:
    # Makes sure that AMLT-based runnings works
    amlt_data_path = os.environ.get('AMLT_DATA_DIR', '')
    amlt_output_path = os.environ.get('AMLT_OUTPUT_DIR', '')

    if args.data_from_amlt:
        args.vocab_path = os.path.join(amlt_data_path, args.vocab_path)
        args.input_file_path = os.path.join(amlt_data_path, args.input_file_path)
    if args.model_from_amlt:
        args.model_path = os.path.join(os.path.dirname(amlt_output_path), args.model_path)
    if args.output_to_amlt:
        args.default_path = os.path.join(amlt_output_path, args.default_path)

    del args.data_from_amlt
    del args.model_from_amlt
    del args.output_to_amlt

    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Score models with Text Predict.')

    paths = parser.add_argument_group('Scoring paths')
    paths.add_argument('--default_path',
                        type=str,
                        default='~/logdir',
                        help='Path to the default folder used to save outputs.')

    paths.add_argument('--model_path',
                        type=str,
                        default=None,
                        help='Path to the model to be loaded.')

    paths.add_argument('--vocab_path',
                        type=str,
                        default=None,
                        help='Path to the vocabulary to be loaded.')

    paths.add_argument('--input_file_path',
                        type=str,
                        default=None,
                        help='Path to the input file to be scored.')

    score = parser.add_argument_group('Scoring types')
    score.add_argument('--input_file_type',
                       type=str,
                       default='smartcompose',
                       choices=['smartcompose', 'text'],
                       help='Type of file to be scored.')

    score.add_argument('--model_type',
                       type=str,
                       default='mem_transformer',
                       choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                       help='Type of model to be scored.')

    score.add_argument('--score_type',
                       type=str,
                       default='torch',
                       choices=['onnx', 'torch'],
                       help='Type of scoring to be used.')

    hyperparameters = parser.add_argument_group('Scoring hyperparameters')
    hyperparameters.add_argument('--save_step',
                                 type=int,
                                 default=100000,
                                 help='Amount of steps to save the predictions.')

    hyperparameters.add_argument('--min_score',
                                 type=float,
                                 default=1.0,
                                 help='Minimum score used within the model.')

    hyperparameters.add_argument('--max_score',
                                 type=float,
                                 default=5.0,
                                 help='Maximum score used within the model.')

    hyperparameters.add_argument('--score_step',
                                 type=float,
                                 default=0.1,
                                 help='Step of the score used within the model.')

    hyperparameters.add_argument('--expected_match_rate',
                                 type=float,
                                 default=0.5,
                                 help='Expected match rate to score the model.')

    hyperparameters.add_argument('--current_paragraph_only',
                                 action='store_true',
                                 help='Uses only current paragraph to score the model.')

    hyperparameters.add_argument('--max_body_len',
                                 type=int,
                                 default=10000,
                                 help='Maximum length of single sequence.')

    hyperparameters.add_argument('--max_seq_len',
                                 type=int,
                                 default=30,
                                 help='Maximum length of sequences to be used.')

    hyperparameters.add_argument('--min_pred_len',
                                 type=int,
                                 default=6,
                                 help='Minimum length of the predictions.')

    amlt = parser.add_argument_group('AMLT-based triggers')
    amlt.add_argument('--data_from_amlt',
                        action='store_true',
                        help='Whether incoming data is from AMLT.')

    amlt.add_argument('--model_from_amlt',
                        action='store_true',
                        help='Whether incoming model is from AMLT.')

    amlt.add_argument('--output_to_amlt',
                        action='store_true',
                        help='Whether output should go to AMLT.')
                    
    args, _ = parser.parse_known_args()
    args = _check_amlt_paths(args)
    
    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()
    
    # Runs the Text Predict scoring
    run_score(args.default_path,
              args.model_path,
              args.vocab_path,
              args.input_file_path,
              args.input_file_type,
              args.model_type,
              score_type=args.score_type,
              save_step=args.save_step,
              min_score=args.min_score,
              max_score=args.max_score,
              score_step=args.score_step,
              expected_match_rate=args.expected_match_rate,
              current_paragraph_only=args.current_paragraph_only,
              max_body_len=args.max_body_len,
              max_seq_len=args.max_seq_len,
              min_pred_len=args.min_pred_len)
