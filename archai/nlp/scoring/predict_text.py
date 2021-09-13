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

def predict_console(text_prediction):
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
            start = time.time()
            (best_prediction, predictions) = text_prediction.predict_full(line)
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


def load_text_file(in_filepath:str)->List[OrderedDict]:
    """Load text file and convert it to SmartCompose file format."""
    logging.info(f"Loading text file from {in_filepath}")
    lines = []
    with open(in_filepath) as f:
        lines = f.readlines()

    positions = []
    for line_id, line in enumerate(lines):
        line = line.rstrip()
        for char_id in range(len(line)):
            pos = OrderedDict({
                "UniqueId": f"{line_id}-{char_id}",
                "Body": line[:char_id],
                "BodyContinued": line[char_id:]
                })
            positions.append(pos)
    return positions

def load_smart_compose_file(in_filepath:str)->List[OrderedDict]:
    """Load SmartCompose .json file format."""
    logging.info(f"Loading smartcompose file from {in_filepath}")
    lines = []
    with open(in_filepath) as f:
        lines = f.readlines()
    positions = [json.loads(line) for line in lines]
    return positions

def predict_smartcompose(text_prediction, positions, max_line_len:int, min_score:float, save_step:int, out_filepath:str):
    """Process dictionary loaded from SmartCompose .json file format."""
    results = []
    for pos in positions:
        start_time = time.time()
        text = pos['Body']
        if len(text) > max_line_len:
            text = pos['Body'][(-1*max_line_len):] # Truncate
            text = text[text.find(' '):]                # Remove partial token
        prediction = text_prediction.predict(text)
        end_time = time.time()
        pos['Time'] = int(1000*(end_time - start_time))
        if len(prediction) > 5 and prediction.score() > min_score:
            pos['Suggestions'] = [{
                'Suggestion': prediction.text,
                'Probability': prediction.probability,
                'Score': prediction.score(),
                }]
        else:
            pos['Suggestions'] = []

        pos_str = json.dumps(pos)
        results.append(pos_str + "\n")
        logging.debug(pos_str)

        if len(results) % save_step == 0 or len(results) == len(positions):
            with open(out_filepath, 'w') as f:
                f.write("".join(results))

    return results

def predict_text(model, vocab, in_filetype:str, in_filepath:str, out_filepath:str, min_score=1.0, save_step=100000, max_line_len=10000):
    predictor = TextPredictor(model, vocab)

    if in_filetype == "console":
        predict_console(predictor)
    elif in_filetype == "text":
        positions = load_text_file(in_filepath)
        predict_smartcompose(predictor, positions, max_line_len, min_score, save_step, out_filepath)
    elif in_filetype == "smartcompose":
        positions = load_smart_compose_file(in_filepath)
        predict_smartcompose(predictor, positions, max_line_len, min_score, save_step, out_filepath)
    else:
        raise ValueError(f"Unkown input type '{in_filetype}'")

