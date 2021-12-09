# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scorer entry-points, such as classes, console and methods.
"""

from __future__ import annotations

from archai.nlp.scoring_metrics.text_predictor import TextPredictor

import logging
import re
import time

import numpy as np

import pandas as pd

from archai.nlp.scoring_metrics.sequence import TextPredictionSequence
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase

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


def score(model, vocab:VocabBase, in_filetype:str,
                 in_filepath:str='', out_filepath:str='', score_output_dir:str='',
                 save_step=100000, min_score=1.0, max_score=5.0, score_step=0.1,
                 expected_match_rate=0.5, # Match point to estimate parameters at
                 current_paragraph_only=False, # Truncate the body to current paragraph only (remove anything before new line
                 do_scoring=True, max_body_len=10000, min_pred_len=6):
    predictor = TextPredictor(model, vocab)
    predictor.MAX_INPUT_TEXT_LEN = max_body_len

    if in_filetype == "console":
        predict_console(predictor)
    elif in_filetype == "text" or in_filetype == "smartcompose":
        assert in_filepath, "in_filepath must be provided"
        seq = TextPredictionSequence.from_file(in_filepath, in_filetype, predictor,
            save_step=save_step, min_score=min_score, current_paragraph_only=current_paragraph_only,
            min_pred_len=min_pred_len
            )
        # seq.MAX_BODY_LEN = max_body_len # Doesn't play well with BOS token
        seq.predict(out_filepath)
        seq.save(out_filepath)
        if do_scoring:
            min_scores = np.arange(min_score, max_score, score_step).tolist()
            seq.score(min_scores, expected_match_rate)
            seq.save_all(score_output_dir, predict_file=None)
    else:
        raise ValueError(f"Unkown input type '{in_filetype}'")


# sample command lines
# tp_predict_text --type smartcompose --model_type swiftkey --model lm+/model_opset12_quant.onnx --tokenizer_type swiftkey --tokenizer tokenizer/ --min_score 2 --input ~/Swiftkey/data/Eval/GSuiteCompete/GSuiteCompete10pc.ljson --output ./GSuiteCompete10pc.ljson --score

# tp_predict_text --type console --model_type transformers --model distilgpt2 --tokenizer_type transformers --tokenizer gpt2

# python scratch/toreli/nlxpy/nlxpy/cli/tp_predict_text.py --type smartcompose --model_type transformers --model distilgpt2 --tokenizer_type transformers --tokenizer gpt2 --input /home/caiocesart/dataroot/metrics/GSuiteCompete10pc.ljson --score

# tp_predict_text --verbose --type console --model_type transformers --model distilgpt2 --tokenizer_type transformers --tokenizer gpt2