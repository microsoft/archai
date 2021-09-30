# Copyright (c) Microsoft Corporation

from __future__ import annotations

import copy
import functools
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Tuple

from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.scoring.vocab_wrapper import VocabWrapper
from archai.nlp.scoring.model_wrapper import ModelWrapper
from archai.nlp.scoring.scoring_utils import WORD_TOKEN_SEPARATOR_SET

class Prediction:
    """Represents a single prediction.
    """
    # Constants for P(Accept) calculation
    a1 = 0.04218
    b1 = -0.1933

    # Penalties for calculating upper case score
    UPPER_PROB_PENALTY = 0.1
    UPPER_SCORE_PENALTY = 0.5
#    UPPER_PROB_PENALTY = 0.0
#    UPPER_SCORE_PENALTY = 0.0

    def __init__(self, text: str, probability: float, predictor: TextPredictor = None,
                 input_ids: tuple = None, token_ids: tuple = None, complete: bool = None, match: bool = None, score: float = None): # pylint: disable=too-many-arguments
        self.text = text
        self.probability = probability
        self.predictor = predictor
        self.input_ids = input_ids
        self.token_ids = token_ids
        self.complete = complete
        self.match = match
        self._score = score

    @classmethod
    def empty(cls, predictor: TextPredictor = None) -> Prediction:
        """Generate empty prediction"""
        return Prediction('', 0.0, predictor=predictor, complete=False)

    @classmethod
    def next_prediction(cls, prediction: Prediction, next_text: str, next_probability: float, next_token_id: int) -> Prediction:
        """Generate new prediction given current prediction and the next step of running the model """
        next_prediction = Prediction(
            prediction.text + next_text,
            prediction.probability * next_probability,
            predictor=prediction.predictor,
            input_ids=prediction.input_ids,
            token_ids=prediction.token_ids + (next_token_id,))
        return next_prediction

    def show(self) -> str:
        """What to actually show (predicted text, right-stripped)"""
        return self.text.rstrip()

    def __len__(self) -> int:
        """Length of the prediction"""
        return len(self.show())

    def __str__(self) -> str:
        """Return text value of the prediction"""
        return self.show()

    def __repr__(self) -> str:
        """Return information about the prediction"""
        return f"({self.text}, {self.probability:.5f}, {self.score():.3f})"

    def p_match(self) -> float:
        """P(Match)"""
        return self.probability

    def p_accept(self) -> float:
        """P(Accept)"""
        result = self.probability * self.p_accept_given_match()
        if result < 0:
            return 0.0
        return result

    def p_accept_given_match(self) -> float:
        """P(Accept|Match)"""
        result = self.a1 * len(self) + self.b1
        if result < 0:
            return 0.0
        return result

    def score(self) -> float:
        """Score we optimize on (currently, maximizing # of characters matched)."""
        if not self._score is None:
            return self._score

        a1 = 0.0 # This is to optimize currently on # of chars matched
        b1 = 1.0

        length = len(self)
        score = self.probability*length*(a1*length + b1)
        if self.predictor is None:
            if not self.is_empty():
                logging.info("Prediction environment not available. Ignoring additional parameters needed for score evaluation")
            return score

        upper_token_ids = self.predictor.vocab_wrapper.UPPER_TOKENS.intersection(self.token_ids)
        if len(upper_token_ids) > 0:
            score = (self.probability - Prediction.UPPER_PROB_PENALTY)*length*(a1*length + b1) - Prediction.UPPER_SCORE_PENALTY

        return score

    def char_accepted(self) -> float:
        """Characters saved (P(Accept) * Prediction Length)."""
        return len(self) * self.p_accept()

    def all_ids(self) -> tuple:
        """Combine the input_ids and token_ids into a single list."""
        if self.input_ids is None or self.token_ids is None:
            raise ValueError(f"Unable to determine combined ids for '{self}'.")
        return self.input_ids + self.token_ids

    def length_type(self) -> str:
        pred_len = len(self)
        if pred_len < 6:
            return "0:XS"

        if pred_len < 11:
            return "1:S"

        if pred_len < 16:
            return "2:M"

        return "3:L"

    def word_count(self) -> int:
        """Return number of words in this prediction."""
        if len(self) == 0:
            return 0
        return len(re.findall(r'\s+', self.text.strip())) + 1

    def update_complete(self) -> bool:
        if self.input_ids is None or self.token_ids is None or self.predictor is None:
            raise ValueError(f"Unable to determine if '{self}' ends with a complete word.")
        self.complete = self.predictor.is_complete_word(tuple(self.input_ids + self.token_ids))
        return self.complete

    def is_valid(self) -> bool:
        """The function determines if this is a valid prediction
        (i.e. one that can be shown to the user)

        Returns:
            bool: Returns True if this is a valid prediction (that should be shown to the user)
        """
        if self.predictor is None:
            raise ValueError("Prediction environment not available; Not able to determine if prediction is valid")

        if self.token_ids is None or len(self.token_ids) == 0:
            return False

        if len(self.show()) < self.predictor.MIN_PRED_LEN:
            return False

        if set(self.token_ids) & self.predictor.vocab_wrapper.INVALID_TOKENS:
            return False

        if self.complete is None:
            self.update_complete()

        if not self.complete:
            return False

        # Moved to Prediction
        # upper_token_ids = self.predictor.tokenizer.UPPER_TOKENS.intersection(self.token_ids)
        # if len(upper_token_ids) > 0 and self.probability < self.predictor.MIN_UPPER_PROB_CUTOFF:
        #    return False

        return True

    def is_empty(self):
        """Return True if prediction is empty"""
        return len(self.text) == 0

    def to_odict(self) -> OrderedDict:
        """Return OrderedDict representing the Prediction."""
        return OrderedDict({
            "Text": self.show(),
            "Probability": self.probability,
            "Length": len(self),
            "Complete": self.complete,
            "Match": self.match,
            "PAccept": self.p_accept(),
            "Score": self.score(),
            "CharAccepted": self.char_accepted(),
            "WordCount": self.word_count(),
            "Tokens": self.token_ids,
            })


class TextPredictor:
    """Main class to run text prediction"""

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
    # Token id to attach to the beginning of the sequence
    BOS_TOKEN_ID = None

    def __init__(self, model, vocab:VocabBase):
        self.model = ModelWrapper(model, vocab.token_to_id(' '), )
        self.vocab_wrapper = VocabWrapper(vocab)

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
        next_token_probs = self.model.get_probs(input_ids)
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
        while calc_count < TextPredictor.PREFIX_MAX_NEXT_CANDIDATE_CALC:
            while_start = time.time()
            calc_count += 1
            #
            # Find unexpanded candidate (elems[2] (i.e. extra_token_len) < 0)
            # filtered_list is a list of a type (idxs + [idx], prob*global_prob, extra_token_len)
            idx_with_reminder = next((i for i, elems in enumerate(filtered_list) \
                    if elems[2] < 0 and elems[1] > TextPredictor.PREFIX_MIN_NEXT_CANDIDATE_PROB \
                        and i <= TextPredictor.PREFIX_MAX_NEXT_CANDIDATE_RANK), None)
            if idx_with_reminder is None: # no tokens with reminder that satisfy our condition
                break

            if idx_with_reminder > 0 \
                and filtered_list[0][1]*TextPredictor.PREFIX_MIN_NEXT_CANDIDATE_PROB_FRAC > filtered_list[idx_with_reminder][1]:
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

        probs = self.model.get_probs(input_ids)
        prob_sum = sum([prob for idx, prob in enumerate(probs) if idx in self.vocab_wrapper.WORD_TOKEN_SEPARATOR_IDX])
        #top = [(idx, self.tokenizer[idx], probs[idx], idx in self.tokenizer.WORD_TOKEN_SEPARATOR_IDX) for idx in reversed(np.argsort(probs)[-20:])]
        #logging.debug("is_complete_word:prob_sum: %s; top: %s", prob_sum, top)

        return prob_sum > TextPredictor.COMPLETE_WORD_PROB_THRESHOLD

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
        if self.BOS_TOKEN_ID is not None and is_full_len:
            input_ids = (self.BOS_TOKEN_ID,) + input_ids

        logging.debug("TextPredictor.predict_full: context[-20:]: '%s';" +\
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
            next_token_id, next_prob = self.model.get_top_token_prob(tuple(prediction.all_ids()))
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

        To get good results, you might have to modify the TextPredictor settings to:
MIN_PRED_LEN = 1
MAX_TOKEN_CALC = 3
MIN_PROB_CUTOFF = 0.001
        """
        (_, results) = self.predict_full(text)
        complete_prediction = next((p for p in results if p.complete), Prediction.empty())
        return complete_prediction.show()

