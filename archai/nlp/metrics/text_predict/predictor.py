# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The actual predicting class (Predictor) for Text Predict.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch


from archai.nlp.datasets.tokenizer_utils.special_token_enum import SpecialTokenEnum
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.metrics.text_predict.prediction import Prediction, TextPredictionSequence
from archai.nlp.metrics.text_predict.text_predict_utils import WORD_TOKEN_SEPARATOR_SET, get_settings
from archai.nlp.metrics.text_predict.wrappers.model_wrapper import ModelWrapper
from archai.nlp.metrics.text_predict.wrappers.vocab_wrapper import VocabWrapper


class Predictor:
    """Runs the text predict pipeline.
    
    """

    # Minimum length of the prediction
    MIN_PRED_LEN = 6

    # Max number of forward calculations to perform
    # (this excludes the calculations at the prefix level)
    MAX_TOKEN_CALC = 6

    # Total probability cutoff at which to terminate the calculation early
    MIN_PROB_CUTOFF = 0.10

    # Constants related to determining correct prefix
    # Max rank of the unexpanded token prime for expansion
    PREFIX_MAX_NEXT_CANDIDATE_RANK = 10

    # Minimum probability of the unexpanded candidate as a fraction of the first candidate (if first is expanded)
    # i.e. if the first candidate is expanded and has prob of 1e-3,
    # the next unexpanded candidate should have prob >= 1e-3*PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC = 5*e-5
    # Note: Candidate has prefix expanded if the input_ids cover the entire text of the prefix,
    # e.g. for prefix 'loo' in 'I am loo' the prediction is:
    # [([1097], 0.010012280195951462, 4), ([286], 0.0002638402802404016, -2), ...]
    # Token 1097 corresponds to ' looking', candidate has +4 characters, is expanded
    # Token 286  corresponds to ' l', candidate is two chars under and is not expanded
    PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC = 0.01
    PREFIX_MAX_NEXT_CANDIDATE_CALC = 5
    PREFIX_MIN_NEXT_CANDIDATE_PROB = 1e-7

    # Return empty result if prefix turns out to be longer than this number
    PREFIX_MAX_LEN = 20

    # Maximum number of characters to lookback for probability mask
    PREFIX_MAX_CHAR_LOOKBACK = 20

    # Probability threshold above which the word is considered 'complete'
    COMPLETE_WORD_PROB_THRESHOLD = 0.75

    # Maximum text to process (otherwise it will be truncated)
    MAX_INPUT_TEXT_LEN = 1000000

    def __init__(self, model: torch.nn.Module, vocab: VocabBase) -> None:
        self.model_wrapper = ModelWrapper(model, vocab.token_to_id(' '), model.tgt_len)
        self.vocab_wrapper = VocabWrapper(vocab)
        self.bos_id = vocab.special_token_id(SpecialTokenEnum.BOS)

    def load_tpl_settings(self, file_name: str) -> None:
        with open(file_name) as json_file:
            settings = json.load(json_file)

        self.MIN_PRED_LEN = settings['MinPredictionLength']
        self.MAX_TOKEN_CALC = settings['MaxTokenCalcCount']
        self.MIN_PROB_CUTOFF = settings['MinProbabiltyCutoff']
        self.PREFIX_MAX_NEXT_CANDIDATE_RANK = settings['MaxNextTokenRank']
        self.PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC = settings['MinNextTokenProbabilityFraction']
        self.PREFIX_MAX_NEXT_CANDIDATE_CALC = settings['MaxNextTokenCalc']
        self.PREFIX_MIN_NEXT_CANDIDATE_PROB = settings['MinNextTokenProbability']
        self.COMPLETE_WORD_PROB_THRESHOLD = settings['CompleteWordThreshold']

        print(settings)

    def settings(self) -> Dict[str, Any]:
        settings = get_settings(self)

        return settings

    @staticmethod
    def score(prob: float, length: int, a1: Optional[float] = 0.0, b1: Optional[float] = 1.0):
        return prob * (a1 * length + b1) * length

    @functools.lru_cache(maxsize=1024)
    def filter_next_tokens(self,
                           input_ids: Optional[Tuple[int, ...]] = (),
                           filter_prefix: Optional[str] ='') -> Tuple[int, ...]:
        next_token_probs = self.model_wrapper.get_probs(input_ids)
        filter_prefix_len = len(filter_prefix)

        if filter_prefix_len == 0:
            result = [((idx,), prob, len(self.vocab_wrapper[idx])) for idx, prob in enumerate(next_token_probs)]
        else:
            filter_next_token_ids = self.vocab_wrapper.filter_token_tuple_ids(filter_prefix)
            result = [(tuple_idx, next_token_probs[tuple_idx[0]], len(self.vocab_wrapper[tuple_idx[0]]) - filter_prefix_len) \
                       for tuple_idx in filter_next_token_ids]

        result = tuple(sorted(result, key=lambda x: -x[1]))

        return result

    # Extended version of the filter_next_tokens
    # For caching, at least within the same session; across session - doesn't really matter
    @functools.lru_cache(maxsize=1024)
    def filter_next_tokens_extended(self,
                                    input_ids: Tuple[int, ...],
                                    filter_prefix: Optional[str] = '',
                                    idxs: Optional[List[int]] =None,
                                    global_prob: Optional[float] = 1.0) -> Tuple[int, ...]:
        filtered_next_token = self.filter_next_tokens(input_ids, filter_prefix)
        
        if idxs is None:
            if global_prob != 1.0:
                filtered_next_token = [(idx, prob*global_prob, extra_token_len) for idx, prob, extra_token_len in filtered_next_token]
            else:
                filtered_next_token = copy.copy(filtered_next_token)
        else:
            filtered_next_token = [(idxs + idx, prob*global_prob, extra_token_len) for idx, prob, extra_token_len in filtered_next_token]

        return filtered_next_token

    def predict_word_prefix(self,
                            input_ids: Tuple[int, ...],
                            prefix: str,
                            debug: Optional[bool] = False) -> Prediction:
        if len(prefix) > self.PREFIX_MAX_LEN:
            logging.debug('predict_word_prefix: prefix: `%s` of length %s is longer than %s chars', prefix, len(prefix), self.PREFIX_MAX_LEN)

            return Prediction.empty()

        start_time = time.time()
        filtered_list = list(self.filter_next_tokens_extended(input_ids, prefix))
        
        logging.debug('predict_word_prefix: prefix: `%s`: filtered_list[:5] %s elem: %s; %.3f ms', prefix, len(filtered_list), filtered_list[:5], 1000*(time.time() - start_time))

        # TPL/C++: Do one or both of:
        # 1) introduce latency constraint (i.e. no more than 20 ms total for this part of the code after the first pass (simple?)
        # 2) Make calc_count only for the calculations that did not hit cache (i.e. ones that take time) (a bit more coding?)
        calc_count = 0
        while calc_count < Predictor.PREFIX_MAX_NEXT_CANDIDATE_CALC:
            while_start = time.time()
            calc_count += 1

            # Find unexpanded candidate (elems[2] (i.e. extra_token_len) < 0)
            # filtered_list is a list of a type (idxs + [idx], prob*global_prob, extra_token_len)
            idx_with_reminder = next((i for i, elems in enumerate(filtered_list) \
                    if elems[2] < 0 and elems[1] > Predictor.PREFIX_MIN_NEXT_CANDIDATE_PROB \
                        and i <= Predictor.PREFIX_MAX_NEXT_CANDIDATE_RANK), None)
            
            # No tokens with reminder that satisfy our condition
            if idx_with_reminder is None:
                break

            if idx_with_reminder > 0 and filtered_list[0][1]*Predictor.PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC > filtered_list[idx_with_reminder][1]:
                logging.debug('Ratio: %s', filtered_list[0][1]/filtered_list[idx_with_reminder][1])
                break

            idxs, prob, filtered_length = filtered_list.pop(idx_with_reminder)
            token_reminder = prefix[filtered_length:]

            filtered_token_prob_with_reminder = self.filter_next_tokens_extended(tuple(input_ids + idxs), token_reminder, tuple(idxs), prob)
            filtered_list.extend(filtered_token_prob_with_reminder)
            filtered_list = sorted(filtered_list, key=lambda x: -x[1])

            if logging.getLogger().level == logging.DEBUG:
                prob_sum = sum([prob for _, prob, _ in filtered_list])
                filtered_list_cond = [(token, prob, score, prob / prob_sum) for  token, prob, score in filtered_list[:5]]

                logging.debug('#%d: %.3f ms: %.3g %s', calc_count, 100 * (time.time() - while_start), prob_sum, filtered_list_cond)

        prediction = Prediction.empty()

        # If empty or first suggestion doesn't complete the token,
        # don't go in (i.e. maintain empty result)
        if len(filtered_list) > 0 and filtered_list[0][2] >= 0:
            prob_sum = sum([prob for _, prob, _ in filtered_list])
            idxs, prob, filtered_length = filtered_list[0]

            pred_text = self.vocab_wrapper.decode(idxs)[len(prefix):]
            prediction = Prediction(pred_text, prob/prob_sum, predictor=self, input_ids=input_ids, token_ids=idxs)
            
            logging.debug('predict_word_prefix: prefix: `%s`: # calcs: %s; time: %.3f ms', prefix, calc_count, 1000*(time.time() - start_time))

        if debug:
            prediction.calc_count = calc_count
            prediction.filtered_list = filtered_list

        return prediction

    @functools.lru_cache(maxsize=1024)
    def is_complete_word(self, input_ids: Tuple[int, ...]) -> bool:
        if len(input_ids) > 0 and self.vocab_wrapper[input_ids[-1]][-1] in WORD_TOKEN_SEPARATOR_SET:
            return True

        probs = self.model_wrapper.get_probs(input_ids)
        prob_sum = sum([prob for idx, prob in enumerate(probs) if idx in self.vocab_wrapper.WORD_TOKEN_SEPARATOR_IDX])

        return prob_sum > Predictor.COMPLETE_WORD_PROB_THRESHOLD

    def truncate_text(self, text: str) -> str:
        if len(text) > self.MAX_INPUT_TEXT_LEN:
            text = text[-self.MAX_INPUT_TEXT_LEN:]
            text = text[text.find(' '):]

        return text

    def predict_full(self, text: str) -> Tuple[Prediction, list]:
        start_time = time.time()

        trunc_text = self.truncate_text(text)
        is_full_len = len(text) == len(trunc_text)

        clean_trunc_text = self.vocab_wrapper.clean(trunc_text, add_bos_text=is_full_len)
        context, prefix = self.vocab_wrapper.find_context_prefix(clean_trunc_text)

        input_ids = tuple(self.vocab_wrapper.encode(context))
        if self.bos_id is not None and is_full_len:
            input_ids = (self.bos_id,) + input_ids

        logging.debug('Predictor.predict_full: context[-20:]: `%s`; ' + \
                      'input_ids[-5:]: %s; prefix: `%s`; time: %.3f ms', context[-20:], input_ids[-5:], prefix, 1000 * (time.time() - start_time))

        # Find the ids corresponding to the prefix first
        prediction = self.predict_word_prefix(input_ids, prefix)

        # Failed to determine the prefix composition
        if prediction.probability == 0.0:
            return (Prediction.empty(), [])

        predictions = [prediction]

        if prediction.is_valid():
            best_prediction = prediction
        else:
            best_prediction = Prediction.empty()

        total_prob = prediction.probability
        calc_count = 0

        while total_prob > self.MIN_PROB_CUTOFF and calc_count < self.MAX_TOKEN_CALC:
            calc_count += 1

            next_token_id, next_prob = self.model_wrapper.get_top_token_prob(tuple(prediction.all_ids()))
            next_text = self.vocab_wrapper.decode([next_token_id])

            prediction = Prediction.next_prediction(prediction, next_text, next_prob, next_token_id)
            prediction.update_complete()

            total_prob = prediction.probability
            predictions.append(prediction)

            if len(prediction) >= self.MIN_PRED_LEN and prediction.is_valid() and prediction.score() >= best_prediction.score() and prediction.probability > self.MIN_PROB_CUTOFF:
                best_prediction = prediction

        logging.debug('Predictions: %s', predictions)

        if len(best_prediction) >= self.MIN_PRED_LEN and best_prediction.is_valid():
            return (best_prediction, predictions)

        return (Prediction.empty(), predictions)

    def predict(self, text: str) -> Prediction:
        (best_result, _) = self.predict_full(text)

        return best_result

    def predict_complete_word(self, text: str) -> str:
        (_, results) = self.predict_full(text)

        complete_prediction = next((p for p in results if p.complete), Prediction.empty())

        return complete_prediction.show()


def predict_console(predictor: Predictor) -> None:
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

                predictor_param_names = [name for name in Predictor.__dict__ if isinstance(Predictor.__dict__[name], int)]
                if param in predictor_param_names:
                    predictor.__dict__[param] = int(value)

                for name in predictor_param_names:
                    value = predictor.__dict__[name] if name in predictor.__dict__ else Predictor.__dict__[name]
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


def score(default_path: str,
          model_path: str,
          vocab_path: str,
          with_onnx: Optional[bool] = False,
          min_score: Optional[float] = 1.0,
          max_score: Optional[float] = 5.0,
          score_step: Optional[float] = 0.1,
          expected_match_rate: Optional[float] = 0.5,
          current_paragraph_only: Optional[bool] = False,
          max_body_len: Optional[int] = 10000,
          min_pred_len: Optional[int] = 6):
        
    #
    model = None
    vocab = None

    #
    predictor = Predictor(model, vocab)
    predictor.MAX_INPUT_TEXT_LEN = max_body_len

    #
    seq = TextPredictionSequence.from_file(in_filepath,
                                           in_filetype,
                                           predictor,
                                           save_step=save_step,
                                           min_score=min_score,
                                           current_paragraph_only=current_paragraph_only,
                                           min_pred_len=min_pred_len)

    #
    seq.predict(out_filepath)
    seq.save(out_filepath)

    min_scores = np.arange(min_score, max_score, score_step).tolist()
    seq.score(min_scores, expected_match_rate)
    seq.save_all(score_output_dir, predict_file=None)
