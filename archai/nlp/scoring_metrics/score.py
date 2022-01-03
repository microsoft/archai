# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scorer entry-points, such as classes, console and methods.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np
import pandas as pd

from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.scoring_metrics.model_wrapper import ModelWrapper
from archai.nlp.scoring_metrics.sequence import TextPredictionSequence
from archai.nlp.scoring_metrics.text_predictor import TextPredictor


def predict_console(predictor: TextPredictor) -> None:
    """Console application that shows predictions.
    
    Args:
        predictor: Instance of the predictor.

    """

    logging.info('Launching console')

    PROMPT = '> '
    START_MSG = 'Press CTRL-D or type `exit` to exit.'

    print(START_MSG)

    try:
        while True:
            line = input(PROMPT)

            if line.strip() == 'exit':
                break

            if line[0:4] == 'set ':
                try:
                    _, param, value = line.strip().split(maxsplit=3)
                except:
                    logging.warning('Could not split `%s` into keyword, param, value', line)

                    param = ''
                    value = 0

                predictor_param_names = [name for name in TextPredictor.__dict__ if isinstance(TextPredictor.__dict__[name], int)]
                if param in predictor_param_names:
                    predictor.__dict__[param] = int(value)

                for name in predictor_param_names:
                    value = predictor.__dict__[name] if name in predictor.__dict__ else TextPredictor.__dict__[name]
                    print(f'{name:30s}\t{value}')

                continue

            line = re.sub(r'\\n', r'\n', line)

            start = time.time()
            (best_prediction, predictions) = predictor.predict_full(line)

            msg = f'Prediction  : {best_prediction.text}\n'
            msg += f'P(Match)    : {best_prediction.p_match():.5f}\n'
            msg += f'CharAccepted: {best_prediction.char_accepted():.5f}\n'
            msg += f'Score       : {best_prediction.score():.5f}\n'
            msg += f'Time (ms)   : {1000*(time.time() - start):.3f}'

            print(msg)

            preds_dict = [p.to_odict() for p in predictions]
            df = pd.DataFrame(preds_dict)

            if 'Match' in df:
                df.drop(['Match'], axis=1)

            print(df)

    except (EOFError, KeyboardInterrupt):
        print('Exiting...')


def score(model: ModelWrapper,
          vocab: VocabBase,
          in_filetype: str,
          in_filepath: Optional[str] = '',
          out_filepath: Optional[str] = '',
          score_output_dir: Optional[str] = '',
          save_step: Optional[int] = 100000,
          min_score: Optional[float] = 1.0,
          max_score: Optional[float] = 5.0,
          score_step: Optional[float] = 0.1,
          expected_match_rate: Optional[float] = 0.5,
          current_paragraph_only: Optional[bool] = False,
          do_scoring: Optional[bool] = True,
          max_body_len: Optional[int] = 10000,
          min_pred_len: Optional[int] = 6) -> None:
    """Performs the scoring procedure.

    Args:
        model: Instance of a wrapped model.
        vocab: Vocabulary.
        in_filetype: Type of input file.
        in_filepath: Path to the input file.
        out_filepath: Path to the output file.
        score_output_dir: Folder to save scoring outputs.
        save_step: Number of steps to save.
        min_score: Mininum score for the scorer.
        max_score: Maximum score for the scorer.
        score_step: Number of steps to score.
        expected_match_rate: Approximate value for the matching rate (parameter learning estimation).
        current_paragraph_only: Truncates the body to current paragraph only (remove anything before new line).
        do_scoring: Whether scoring should be performed or not.
        max_body_len: Maximum length of the body.
        min_pred_len: Minimum length of the prediction.

    """

    predictor = TextPredictor(model, vocab)
    predictor.MAX_INPUT_TEXT_LEN = max_body_len

    if in_filetype == 'console':
        predict_console(predictor)

    elif in_filetype == 'text' or in_filetype == 'smartcompose':
        assert in_filepath, 'in_filepath must be provided'

        seq = TextPredictionSequence.from_file(in_filepath,
                                               in_filetype,
                                               predictor,
                                               save_step=save_step,
                                               min_score=min_score,
                                               current_paragraph_only=current_paragraph_only,
                                               min_pred_len=min_pred_len)

        seq.predict(out_filepath)
        seq.save(out_filepath)

        if do_scoring:
            min_scores = np.arange(min_score, max_score, score_step).tolist()
            seq.score(min_scores, expected_match_rate)
            seq.save_all(score_output_dir, predict_file=None)

    else:
        raise ValueError(f'Unkown input type `{in_filetype}`')
