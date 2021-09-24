#!/usr/bin/env python

# Copyright (c) Microsoft Corporation

from __future__ import annotations
from archai.nlp.scoring.text_predictor import TextPredictor # pylint: disable=misplaced-future
from archai.nlp.scoring.scoring_utils import WORD_TOKEN_SEPARATOR_SET

import argparse
import json
import logging
import re
import readline  # pylint: disable=unused-import
import time
from collections import OrderedDict
from typing import List

import pandas as pd
import torch

def predict_console(predictor:TextPredictor):
    """Console application showing predictions.
    """
    logging.info("Launching console")

    PROMPT = "> "
    START_MSG = "Press CTRL-D or type 'exit' to exit."

    print(START_MSG)
    try:
        while True:
            line = input(PROMPT)
            if line.strip() == "exit":
                break

            if line[0:4] == "set ":
                try:
                    _, param, value = line.strip().split(maxsplit=3)
                except:
                    logging.warning("Could not split '%s' into keyword, param, value", line)
                    param = ""
                    value = 0

                predictor_param_names = [name for name in TextPredictor.__dict__ if isinstance(TextPredictor.__dict__[name], int)]
                if param in predictor_param_names:
                    predictor.__dict__[param] = int(value)

                for name in predictor_param_names:
                    value = predictor.__dict__[name] if name in predictor.__dict__ else TextPredictor.__dict__[name]
                    print(f"{name:30s}\t{value}")

                continue

            line = re.sub(r"\\n", r"\n", line)

            start = time.time()
            (best_prediction, predictions) = predictor.predict_full(line)
            msg = f"Prediction  : {best_prediction.text}\n"
            msg += f"P(Match)    : {best_prediction.p_match():.5f}\n"
            msg += f"CharAccepted: {best_prediction.char_accepted():.5f}\n"
            msg += f"Score       : {best_prediction.score():.5f}\n"
            msg += f"Time (ms)   : {1000*(time.time() - start):.3f}"
            print(msg)
            preds_dict = [p.to_odict() for p in predictions]
            df = pd.DataFrame(preds_dict)
            if 'Match' in df:
                df.drop(['Match'], axis=1)
            print(df)
    except (EOFError, KeyboardInterrupt):
        print("Exiting...")


def predict_text(model, vocab, in_filetype:str, in_filepath:str, out_filepath:str, min_score=1.0, save_step=100000, max_body_len=10000):
    predictor = TextPredictor(model, vocab)

    if in_filetype == "console":
        predict_console(predictor)
    elif in_filetype == "text" or in_filetype == "smartcompose":
        seq = TextPredictionSequence.from_file(in_filepath, in_filetype, predictor)
        # seq.MAX_BODY_LEN = max_body_len # Doesn't play well with BOS token
        seq.SAVE_STEP = save_step
        seq.MIN_SCORE = min_score
        seq.CURRENT_PARAGRAPH_ONLY = current_paragraph_only
        seq.predict(output)
        seq.save(output)
        if score:
            min_scores = np.arange(min_score, max_score, score_step).tolist()
            seq.score(min_scores, expected_match_rate)
            seq.save_all(score_output_dir, predict_file=None)
    else:
        raise ValueError(f"Unkown input type '{in_filetype}'")

