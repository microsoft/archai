# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import re
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from torch import nn

from archai.nlp.datasets.tokenizer_utils.special_token_enum import \
    SpecialTokenEnum
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.metrics.model_wrapper import ModelWrapper
from archai.nlp.metrics.text_predict.prediction import Prediction
from archai.nlp.metrics.text_predict.prediction_sequence import \
    TextPredictionSequence
from archai.nlp.metrics.text_predict.wrappers.model_wrapper import ModelWrapper
from archai.nlp.metrics.text_predict.wrappers.vocab_wrapper import VocabWrapper
from archai.nlp.metrics.text_predict_utils import (WORD_TOKEN_SEPARATOR_SET,
                                                   get_settings)
from archai.nlp.metrics.text_prediction import TextPredictor


class Predictor:
    """Runs the text predict pipeline.
    
    """

    # Fundamental constants that control TextPrediction
    # Minimum length of the prediction
    MIN_PRED_LEN = 6

    # Max number of forward calculations to perform
    # (this excludes the calculations at the prefix level)
    MAX_TOKEN_CALC = 6

    # Total probability cutoff at which to terminate the calculation early
    MIN_PROB_CUTOFF = 0.10

    # Probability cutoff for suggestions with uppercase letters
    # MIN_UPPER_PROB_CUTOFF = 0.20 # Moved to Prediction

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
    MAX_INPUT_TEXT_LEN = 1_000_000

    def __init__(self, model:nn.Module, vocab:VocabBase):
        self.model_wrapper = ModelWrapper(model, vocab.token_to_id(' '), model.tgt_len)
        self.vocab_wrapper = VocabWrapper(vocab)
        self.bos_id = vocab.special_token_id(SpecialTokenEnum.BOS)

    def load_tpl_settings(self, file_name):
        """Load settings from a file like this:
{
    "VocabFile": "vocab2.txt",
    "MergesFile": "merges2.txt",
    "ModelFile": "model_opset12_quant.ort",
    "InputStateName": "state",
    "InputWordIndexName": "word_indices",
    "OutputScoreName": "83",
    "OutputStateName": "65",
    "StateSize": 512,
    "NumTokensToUse": 30,
    "MinPredictionLength": 6,
    "MaxTokenCalcCount": 6,
    "MinProbabiltyCutoff": 0.10,
    "MaxNextTokenRank": 5,
    "MinNextTokenProbabilityFraction": 0.05,
    "MaxNextTokenCalc": 5,
    "MinNextTokenProbability": 0.0000001,
    "CompleteWordThreshold": 0.75,
    "ScoreWeightA": 0.0,
    "ScoreWeightB": 1.0,
    "MinScore": 1.0
}
        """
        with open(file_name) as json_file:
            settings = json.load(json_file)
        self.MIN_PRED_LEN = settings["MinPredictionLength"] # pylint: disable=invalid-name
        self.MAX_TOKEN_CALC = settings["MaxTokenCalcCount"] # pylint: disable=invalid-name
        self.MIN_PROB_CUTOFF = settings["MinProbabiltyCutoff"] # pylint: disable=invalid-name
        self.PREFIX_MAX_NEXT_CANDIDATE_RANK = settings["MaxNextTokenRank"] # pylint: disable=invalid-name
        self.PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC = settings["MinNextTokenProbabilityFraction"] # pylint: disable=invalid-name
        self.PREFIX_MAX_NEXT_CANDIDATE_CALC = settings["MaxNextTokenCalc"] # pylint: disable=invalid-name
        self.PREFIX_MIN_NEXT_CANDIDATE_PROB = settings["MinNextTokenProbability"] # pylint: disable=invalid-name
        self.COMPLETE_WORD_PROB_THRESHOLD = settings["CompleteWordThreshold"] # pylint: disable=invalid-name
        print(settings)

    def settings(self) -> dict:
        settings = get_settings(self)
        return settings

    @staticmethod
    def score(prob, length, a1=0.0, b1=1.0):
        """Function to calculate the score for the trigger mechanism."""
        return prob*(a1*length + b1)*length

    @functools.lru_cache(maxsize=1024)
    def filter_next_tokens(self, input_ids=(), filter_prefix='') -> tuple:
        """Returns distribution of the probabilities filtered by the prefix.
           It is sorted in a descending order by the probability of occurrence

           Specifically, the returned result is a tuple with:
            token_id, probability, filtered_length
        E.g.: filter_next_token((), 'Congrat')
            [(34, 0.007192195858806372, -6),
            (3103, 0.0006149907712824643, -4),
            (7222, 0.00029342802008613944, -5),
            (18649, 1.9916371456929483e-05, -3),
            (45048, 3.0129864171613008e-05, 8)]
        (i.e. token '34' is 'C', which means that 6 characters still were not filtered away)
        """
        # start_time = time.time()
        next_token_probs = self.model_wrapper.get_probs(input_ids)
        filter_prefix_len = len(filter_prefix)
        if filter_prefix_len == 0:
            result = [((idx,), prob, len(self.vocab_wrapper[idx])) for idx, prob in enumerate(next_token_probs)]
        else:
            #filter_next_token_ids = self.tokenizer.filter_token_tuple_ids(filter_prefix, self.PREFIX_MAX_CHAR_LOOKBACK) # This is slow (sometimes ms)
            filter_next_token_ids = self.vocab_wrapper.filter_token_tuple_ids(filter_prefix) # This is slow (sometimes ms)
            # logging.debug("filter_next_tokens0: input_ids: %s; filter_prefix: '%s'; len(filter_next_token_ids): %s, time: %.3f ms", input_ids, filter_prefix, len(filter_next_token_ids), 1000*(time.time() - start_time))
#            try:
            result = [(tuple_idx, next_token_probs[tuple_idx[0]], len(self.vocab_wrapper[tuple_idx[0]]) - filter_prefix_len) \
                for tuple_idx in filter_next_token_ids]
#            except:
#                print(f"EXCEPT: {input_ids} {len(next_token_probs)}")
#                print(filter_next_token_ids)
#                print(result)

        # logging.debug("filter_next_tokens1: input_ids: %s; filter_prefix: '%s'; len(result): %s, time: %.3f ms", input_ids, filter_prefix, len(result), 1000*(time.time() - start_time))
        result = tuple(sorted(result, key=lambda x: -x[1]))
        # logging.debug("filter_next_tokens2: input_ids: %s; filter_prefix: '%s'; len(result): %s, time: %.3f ms", input_ids, filter_prefix, len(result), 1000*(time.time() - start_time))
        # if len(input_ids) == 2 and input_ids[0] == 200 and input_ids[1] == 6542:
        #     print(f"filter_next_tokens: {input_ids} '{filter_prefix}' len(result) = {len(result)} {result[:5]} {next_token_probs[334:337]}")

        return result

    # Extended version of the filter_next_tokens
    # For caching, at least within the same session; across session - doesn't really matter
    # @deepcopy_lru_cache(maxsize=1024)
    @functools.lru_cache(maxsize=1024)
    def filter_next_tokens_extended(self, input_ids: tuple, filter_prefix='', idxs=None, global_prob=1.0) -> tuple:
        """Helper function for filter_next_token
        """
        # Note that it is sorted descending according to probability (first element has the largest prob)
        # start_time = time.time()
        filtered_next_token = self.filter_next_tokens(input_ids, filter_prefix)
        # if len(input_ids) == 2 and input_ids[0] == 200 and input_ids[1] == 6542:
        # print(f"filter_next_tokens_extended: {input_ids} {filter_prefix} {idxs} {global_prob} {len(filtered_next_token)} {filtered_next_token[:5]}")
        # logging.debug("filter_next_tokens_list1: input_ids: %s; filter_prefix: '%s'; idx: %s, len(filtered_next_token): %s, time: %.3f ms", input_ids, filter_prefix, idxs, len(filtered_next_token), 1000*(time.time() - start_time))
        if idxs is None:
            if global_prob != 1.0:
                filtered_next_token = [(idx, prob*global_prob, extra_token_len) for idx, prob, extra_token_len in filtered_next_token]
            else:
                filtered_next_token = copy.copy(filtered_next_token)
        else:
            filtered_next_token = [(idxs + idx, prob*global_prob, extra_token_len) for idx, prob, extra_token_len in filtered_next_token]

        # logging.debug("filter_next_tokens_list2: input_ids: %s; filter_prefix: '%s'; idx: %s, len(filtered_next_token): %s, time: %.3f ms", input_ids, filter_prefix, idxs, len(filtered_next_token), 1000*(time.time() - start_time))
        return filtered_next_token

    def predict_word_prefix(self, input_ids: tuple, prefix: str, debug: bool = False) -> Prediction:
        """Provide the best guess for the completion of the prefix.
        Returns a tuple with list of a sort: (token ids that complete the token, probability of completion)
        e.g.:
In [67]: predict_word_prefix([], 'Congra')
Out[67]: ([45048], 0.7450297743888145)
In [68]: idx2token[45048]
Out[68]: 'Congratulations'
During the expansion two things are happening:
    1) Find the most likely token (first in a list sorted by probabilities)
    2) Reduce the total probability mass by the expansion process
Run it until one of the conditions is met:
    1) At most PREFIX_MAX_NEXT_CANDIDATE_CALC times
    2) No more tokens to expand in top PREFIX_MAX_NEXT_CANDIDATE_RANK ranked tokens
    3) if the probability mass in the first token is greater than PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC of probability of the next token to expand
Note:
If you run algortihm too many times, it just gets slower
If you don't run it enough, you might not end up with any prefix or prefix will have too low probability
        """
        if len(prefix) > self.PREFIX_MAX_LEN:
            logging.debug("predict_word_prefix: prefix: '%s' of length %s is longer than %s chars", prefix, len(prefix), self.PREFIX_MAX_LEN)
            return Prediction.empty()

        start_time = time.time()
        filtered_list = list(self.filter_next_tokens_extended(input_ids, prefix))
        # if len(input_ids) == 2 and input_ids[0] == 200 and input_ids[1] == 6542:
        # print(f"filter_next_tokens_extended2: {input_ids} '{prefix}' {len(filtered_list)} {filtered_list[:5]}")

        # filtered_list_orig_len = len(filtered_list)
        logging.debug("predict_word_prefix: prefix: '%s': filtered_list[:5] %s elem: %s; %.3f ms", prefix, len(filtered_list), filtered_list[:5], 1000*(time.time() - start_time))

        calc_count = 0
        # TPL/C++: Do one or both of:
        # 1) introduce latency constraint (i.e. no more than 20 ms total for this part of the code after the first pass (simple?)
        # 2) Make calc_count only for the calculations that did not hit cache (i.e. ones that take time) (a bit more coding?)
        while calc_count < Predictor.PREFIX_MAX_NEXT_CANDIDATE_CALC:
            while_start = time.time()
            calc_count += 1
            #
            # Find unexpanded candidate (elems[2] (i.e. extra_token_len) < 0)
            # filtered_list is a list of a type (idxs + [idx], prob*global_prob, extra_token_len)
            idx_with_reminder = next((i for i, elems in enumerate(filtered_list) \
                    if elems[2] < 0 and elems[1] > Predictor.PREFIX_MIN_NEXT_CANDIDATE_PROB \
                        and i <= Predictor.PREFIX_MAX_NEXT_CANDIDATE_RANK), None)
            if idx_with_reminder is None: # no tokens with reminder that satisfy our condition
                break

            if idx_with_reminder > 0 \
                and filtered_list[0][1]*Predictor.PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC > filtered_list[idx_with_reminder][1]:
                logging.debug("Ratio: %s", filtered_list[0][1]/filtered_list[idx_with_reminder][1])
                break

            idxs, prob, filtered_length = filtered_list.pop(idx_with_reminder)
            token_reminder = prefix[filtered_length:]
            filtered_token_prob_with_reminder = self.filter_next_tokens_extended(tuple(input_ids + idxs), token_reminder, tuple(idxs), prob)
            filtered_list.extend(filtered_token_prob_with_reminder)
            filtered_list = sorted(filtered_list, key=lambda x: -x[1])
            if logging.getLogger().level == logging.DEBUG:
                prob_sum = sum([prob for token, prob, score in filtered_list])
                filtered_list_cond = [(token, prob, score, prob/prob_sum) for  token, prob, score in filtered_list[:5]]
                logging.debug("#%d: %.3f ms: %.3g %s", calc_count, 100*(time.time() - while_start), prob_sum, filtered_list_cond)

        prediction = Prediction.empty()
        # If empty or first suggestion doesn't complete the token, don't go in (i.e. maintain empty result)
        if len(filtered_list) > 0 and filtered_list[0][2] >= 0:
            prob_sum = sum([prob for idxs, prob, filtered_length in filtered_list])
            idxs, prob, filtered_length = filtered_list[0]
            pred_text = self.vocab_wrapper.decode(idxs)[len(prefix):]

            prediction = Prediction(pred_text, prob/prob_sum, predictor=self, input_ids=input_ids, token_ids=idxs)
            logging.debug("predict_word_prefix: prefix: '%s': # calcs: %s; time: %.3f ms", prefix, calc_count, 1000*(time.time() - start_time))

        if debug:
            prediction.calc_count = calc_count # pylint: disable=attribute-defined-outside-init
            prediction.filtered_list = filtered_list # pylint: disable=attribute-defined-outside-init
            # prediction.filtered_list_orig_len = filtered_list_orig_len # pylint: disable=attribute-defined-outside-init

        return prediction


    @functools.lru_cache(maxsize=1024)
    def is_complete_word(self, input_ids: tuple) -> bool:
        """The function checks if the word is complete.

        The word is complete if:
        1) the last character is one of WORD_TOKEN_SEPARATOR
        2) the next token has one of the WORD_TOKEN_SEPARATOR
        as the first character (e.g. space, new line, etc.).
        More specifically, if cumulative probability of such character is > COMPLETE_WORD_PROB_THRESHOLD

        Args:
            input_ids (tuple): indices of tokenized text including the final word

        Returns:
            bool: True/False determining whether the word is 'complete'.
        """
        if len(input_ids) > 0 and self.vocab_wrapper[input_ids[-1]][-1] in WORD_TOKEN_SEPARATOR_SET:
            return True

        probs = self.model_wrapper.get_probs(input_ids)
        prob_sum = sum([prob for idx, prob in enumerate(probs) if idx in self.vocab_wrapper.WORD_TOKEN_SEPARATOR_IDX])
        #top = [(idx, self.tokenizer[idx], probs[idx], idx in self.tokenizer.WORD_TOKEN_SEPARATOR_IDX) for idx in reversed(np.argsort(probs)[-20:])]
        #logging.debug("is_complete_word:prob_sum: %s; top: %s", prob_sum, top)

        return prob_sum > Predictor.COMPLETE_WORD_PROB_THRESHOLD

    def truncate_text(self, text: str) -> str:
        if len(text) > self.MAX_INPUT_TEXT_LEN:
            text = text[-self.MAX_INPUT_TEXT_LEN:] # Truncate
            text = text[text.find(' '):]           # Remove partial token

        return text


    def predict_full(self, text: str) -> Tuple[Prediction, list]:
        """Return most likely predicted text and associated scores."""
        start_time = time.time()
        # TODO: Test truncate / clean text
        trunc_text = self.truncate_text(text)
        is_full_len = len(text) == len(trunc_text)
        clean_trunc_text = self.vocab_wrapper.clean(trunc_text, add_bos_text=is_full_len)

        context, prefix = self.vocab_wrapper.find_context_prefix(clean_trunc_text)

        input_ids = tuple(self.vocab_wrapper.encode(context))
        if self.bos_id is not None and is_full_len:
            input_ids = (self.bos_id,) + input_ids

        logging.debug("Predictor.predict_full: context[-20:]: '%s';" +\
            " input_ids[-5:]: %s; prefix: '%s'; time: %.3f ms", context[-20:], input_ids[-5:], prefix, 1000*(time.time() - start_time))

        # Find the ids corresponding to the prefix first
        prediction = self.predict_word_prefix(input_ids, prefix)
        if prediction.probability == 0.0: # Failed to determine the prefix composition
            return (Prediction.empty(), [])

#        prediction.update_complete()
        predictions = [prediction]
#        if len(prediction) >= self.MIN_PRED_LEN and prediction.is_valid():
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
            if len(prediction) >= self.MIN_PRED_LEN \
                    and prediction.is_valid() \
                    and prediction.score() >= best_prediction.score() \
                    and prediction.probability > self.MIN_PROB_CUTOFF:
                best_prediction = prediction

        logging.debug("Predictions: %s", predictions)

        if len(best_prediction) >= self.MIN_PRED_LEN and best_prediction.is_valid():
            return (best_prediction, predictions)

        return (Prediction.empty(), predictions)

    def predict(self, text: str) -> Prediction:
        """Return predicted text with the highest score (simplified version of predict_full)"""
        (best_result, _) = self.predict_full(text)
        return best_result

    def predict_complete_word(self, text: str) -> str:
        """Return most likely complete word.

        To get good results, you might have to modify the Predictor settings to:
MIN_PRED_LEN = 1
MAX_TOKEN_CALC = 3
MIN_PROB_CUTOFF = 0.001
        """
        (_, results) = self.predict_full(text)
        complete_prediction = next((p for p in results if p.complete), Prediction.empty())
        return complete_prediction.show()



def predict_console(predictor: TextPredictor) -> None:
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
          expected_match_rate: Optional[float] = 0.5, # Match point to estimate parameters at
          current_paragraph_only: Optional[bool] = False, # Truncates the body to current paragraph only (remove anything before new line)
          do_scoring: Optional[bool] =True,
          max_body_len: Optional[int] = 10000,
          min_pred_len: Optional[int] = 6):
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
