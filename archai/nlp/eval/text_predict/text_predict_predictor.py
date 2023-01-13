# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Text Predict-based predictor."""

import copy
import functools
import re
import time
from typing import List, Optional, Tuple, Union

from scipy.interpolate import interp1d
from tqdm import tqdm

from archai.nlp.eval.text_predict.text_predict_model import TextPredictModel
from archai.nlp.eval.text_predict.text_predict_prediction import (
    TextPredictionSequence,
    TextPredictPrediction,
)
from archai.nlp.eval.text_predict.text_predict_tokenizer import (
    SEPARATOR_TOKENS_SET,
    TextPredictTokenizer,
)


class Predictor:
    """Text Predict-based predictor."""

    # Maximum number of forward passes
    MAX_FORWARD_PASS = 6

    # Minimum probability cutoff to calculate early stopping
    MIN_PROB_CUTOFF = 0.10

    # Maximum length, forward passes and number of expansions for prefix
    PREFIX_MAX_LENGTH = 20
    PREFIX_MAX_NEXT_CANDIDATE_FORWARD_PASS = 5
    PREFIX_MAX_NEXT_CANDIDATE_RANK = 10

    # Maximum number of characters to lookback the probability mask
    PREFIX_MAX_CHAR_LOOKBACK = 20

    # If the first candidate is expanded and has prob of 1e-3,
    # the next unexpanded candidate should have
    # prob >= 1e-3 * PREFIX_MIN_NEXT_CANDIDATE_PROB_FRACTION = 5 * e-5
    PREFIX_MIN_NEXT_CANDIDATE_PROB = 1e-7
    PREFIX_MIN_NEXT_CANDIDATE_PROB_FRACTION = 0.01

    # Note: Candidate has prefix expanded if the `input_ids` cover the entire text of the prefix,
    # e.g. for prefix 'loo' in 'I am loo' the prediction is:
    # [([1097], 0.010012280195951462, 4), ([286], 0.0002638402802404016, -2), ...]
    # Token 1097 corresponds to ' looking', candidate has +4 characters, is expanded
    # Token 286  corresponds to ' l', candidate is two chars under and is not expanded
    # Minimum probability of the unexpanded candidate as a fraction of the first candidate.

    # Probability threshold above which the word is considered 'complete'
    COMPLETE_WORD_PROB_THRESHOLD = 0.75

    def __init__(
        self,
        model: TextPredictModel,
        tokenizer: TextPredictTokenizer,
        max_body_length: Optional[int] = 1000000,
        min_pred_length: Optional[int] = 6,
    ) -> None:
        """Initialize predictor.

        Args:
            model: An instance of a Text Predict-based model.
            tokenizer: An instance of a Text Predict-based tokenizer.
            max_body_length: Maximum text to process (otherwise it will be truncated).
            min_pred_length: Minimum length (tokens) of prediction.

        """

        self.model = model
        self.tokenizer = tokenizer

        self.max_body_length = max_body_length
        self.min_pred_length = min_pred_length
        self.bos_token_id = None

    def _truncate_text(self, text: str) -> str:
        """Truncate a text based on the maximium allowed length.

        Args:
            text: The text to truncate.

        Returns:
            The truncated text.

        """

        if len(text) > self.max_body_length:
            text = text[-self.max_body_length :]
            text = text[text.find(" ") :]

        return text

    @functools.lru_cache(maxsize=1024)
    def _check_end_with_complete_word(self, input_ids: Tuple[int, ...]) -> bool:
        """Check if predicted word (set of tokens) is complete according to threshold.

        Args:
            input_ids: The tokens that identify predicted word.

        Returns:
            Whether predicted word is complete or not.

        """

        if len(input_ids) > 0 and self.tokenizer[input_ids[-1]][-1] in SEPARATOR_TOKENS_SET:
            return True

        probs = self.model.get_probs(input_ids)
        probs_sum = sum(
            [prob for idx, prob in enumerate(probs) if idx in self.tokenizer.TOKENIZER_WORD_TOKEN_SEPARATOR]
        )

        return probs_sum > Predictor.COMPLETE_WORD_PROB_THRESHOLD

    def _update_end_with_complete_word(self, prediction: TextPredictPrediction) -> bool:
        """Update whether prediction defines a complete word or not.

        Args:
            prediction: The prediction.

        Returns:
            Whether prediction defines a complete word or not.

        """

        if prediction.input_ids is None or prediction.token_ids is None:
            raise ValueError(f"Unable to determine if `{prediction}` ends with a complete word.")

        prediction.end_with_complete_word = self._check_end_with_complete_word(
            tuple(prediction.input_ids + prediction.token_ids)
        )

        return prediction.end_with_complete_word

    def _check_valid_prediction(self, prediction: TextPredictPrediction) -> bool:
        """Check whether prediction is valid or not.

        Args:
            prediction: The prediction.

        Returns:
            Whether prediction is valid or not.

        """

        if prediction.token_ids is None or len(prediction.token_ids) == 0:
            return False

        if len(str(prediction)) < self.min_pred_length:
            return False

        if set(prediction.token_ids) & self.tokenizer.INVALID_TOKENS:
            return False

        if prediction.end_with_complete_word is None:
            self._update_end_with_complete_word(prediction)

        if not prediction.end_with_complete_word:
            return False

        return True

    @functools.lru_cache(maxsize=1024)
    def _initial_filter_next_tokens(
        self, input_ids: Optional[Tuple[int, ...]] = None, filter_prefix: Optional[str] = ""
    ) -> Tuple[int, ...]:
        """Core computation to predict and filter tokens according to the supplied prefix.

        Args:
            input_ids: The input tokens.
            filter_prefix: The prefix to filter.

        Returns:
            A list of filtered tokens.

        """

        next_token_probs = self.model.get_next_token_probs(input_ids)

        filter_prefix_length = len(filter_prefix)
        if filter_prefix_length == 0:
            filtered_tokens = [((idx,), prob, len(self.tokenizer[idx])) for idx, prob in enumerate(next_token_probs)]
        else:
            filtered_tokens = self.tokenizer.filter_tokens(filter_prefix)
            filtered_tokens = tuple((idx,) for idx in filtered_tokens)
            filtered_tokens = [
                (
                    tuple_idx,
                    next_token_probs[tuple_idx[0]],
                    len(self.tokenizer[tuple_idx[0]]) - filter_prefix_length,
                )
                for tuple_idx in filtered_tokens
            ]

        filtered_tokens = tuple(sorted(filtered_tokens, key=lambda x: -x[1]))

        return filtered_tokens

    @functools.lru_cache(maxsize=1024)
    def _filter_next_tokens(
        self,
        input_ids: Tuple[int, ...],
        filter_prefix: Optional[str] = "",
        idxs: Optional[List[int]] = None,
        global_prob: Optional[float] = 1.0,
    ) -> Tuple[int, ...]:
        """Predict and filter tokens according to the supplied prefix.

        Args:
            input_ids: The input tokens.
            filter_prefix: The prefix to filter.
            idxs: Additional indexes from the expansion procedure.
            global_prob: The global probability of the expansion procedure.

        Returns:
            A list of filtered tokens.

        """

        filtered_tokens = self._initial_filter_next_tokens(input_ids, filter_prefix)

        if idxs is None:
            if global_prob != 1.0:
                filtered_tokens = [
                    (idx, prob * global_prob, extra_token_length) for idx, prob, extra_token_length in filtered_tokens
                ]
            else:
                filtered_tokens = copy.copy(filtered_tokens)
        else:
            filtered_tokens = [
                (idxs + idx, prob * global_prob, extra_token_length)
                for idx, prob, extra_token_length in filtered_tokens
            ]

        return filtered_tokens

    def _find_initial_prediction(self, input_ids: Tuple[int, ...], prefix: str) -> TextPredictPrediction:
        """Predict prefix from a supplied word.

        Args:
            input_ids: The input tokens.
            prefix: The prefix to predict.

        Returns:
            The initial prediction.

        """

        if len(prefix) > self.PREFIX_MAX_LENGTH:
            return TextPredictPrediction.empty()

        # List of (idxs + [idx], prob * global_prob, extra_token_length)
        filtered_tokens = list(self._filter_next_tokens(input_ids, prefix))

        n_forward_pass = 0
        while n_forward_pass < Predictor.PREFIX_MAX_NEXT_CANDIDATE_FORWARD_PASS:
            n_forward_pass += 1

            # Finds unexpanded candidate (token[2] (i.e. extra_token_length) < 0)
            unexpanded_token_idx = next(
                (
                    i
                    for i, token in enumerate(filtered_tokens)
                    if token[2] < 0
                    and token[1] > Predictor.PREFIX_MIN_NEXT_CANDIDATE_PROB
                    and i <= Predictor.PREFIX_MAX_NEXT_CANDIDATE_RANK
                ),
                None,
            )

            if unexpanded_token_idx is None:
                break

            if (
                unexpanded_token_idx > 0
                and filtered_tokens[0][1] * Predictor.PREFIX_MIN_NEXT_CANDIDATE_PROB_FRACTION
                > filtered_tokens[unexpanded_token_idx][1]
            ):
                break

            idxs, prob, filtered_length = filtered_tokens.pop(unexpanded_token_idx)
            unexpanded_token = prefix[filtered_length:]

            filtered_unexpanded_token = self._filter_next_tokens(
                tuple(input_ids + idxs), unexpanded_token, tuple(idxs), prob
            )
            filtered_tokens.extend(filtered_unexpanded_token)
            filtered_tokens = sorted(filtered_tokens, key=lambda x: -x[1])

        prediction = TextPredictPrediction.empty()

        # If empty or first suggestion does not complete the token,
        # do not go in (i.e. maintain empty result)
        if len(filtered_tokens) > 0 and filtered_tokens[0][2] >= 0:
            probs_sum = sum([prob for _, prob, _ in filtered_tokens])
            idxs, prob, filtered_length = filtered_tokens[0]

            text = self.tokenizer.decode(idxs)[len(prefix) :]
            prediction = TextPredictPrediction(text, prob / probs_sum, input_ids=input_ids, token_ids=idxs)

        return prediction

    def _predict(self, text: str) -> TextPredictPrediction:
        """Core computation to perform the prediction pipeline.

        Args:
            text: The text to predict.

        Returns:
            The prediction.

        """

        truncated_text = self._truncate_text(text)
        is_full_length = len(text) == len(truncated_text)

        clean_truncated_text = self.tokenizer.clean_text(truncated_text, add_bos_text=is_full_length)
        context, prefix = self.tokenizer.find_context_and_prefix(clean_truncated_text)

        input_ids = tuple(self.tokenizer.encode(context))
        if self.bos_token_id is not None and is_full_length:
            input_ids = (self.bos_token_id,) + input_ids

        prediction = self._find_initial_prediction(input_ids, prefix)
        if prediction.probability == 0.0:
            return TextPredictPrediction.empty()

        if self._check_valid_prediction(prediction):
            best_prediction = prediction
        else:
            best_prediction = TextPredictPrediction.empty()

        total_prob = prediction.probability
        n_forward_pass = 0
        while total_prob > self.MIN_PROB_CUTOFF and n_forward_pass < self.MAX_FORWARD_PASS:
            n_forward_pass += 1

            next_token_id, next_prob = self.model.get_top_next_token_probs(tuple(prediction.all_ids()))
            next_text = self.tokenizer.decode([next_token_id])

            prediction = TextPredictPrediction.next_prediction(prediction, next_text, next_prob, next_token_id)
            self._update_end_with_complete_word(prediction)

            total_prob = prediction.probability

            if (
                len(prediction) >= self.min_pred_length
                and prediction._check_valid_prediction()
                and prediction.score() >= best_prediction.score()
                and prediction.probability > self.MIN_PROB_CUTOFF
            ):
                best_prediction = prediction

        if len(best_prediction) >= self.min_pred_length and best_prediction._check_valid_prediction():
            return best_prediction

        return TextPredictPrediction.empty()

    def predict(self, sequences: List[TextPredictionSequence], output_file: Optional[str] = None) -> None:
        """Predict a set of sequences.

        Args:
            sequences: List of sequences to predict.
            output_file: Optional output file to write the predictions.

        """

        for i, pos in enumerate(tqdm(sequences.values(), desc="Predicting")):
            start_time = time.time()

            text = pos.body
            if sequences.current_paragraph_only:
                text = re.sub("^(.*\n)", "", text, flags=re.M)
            if len(text) > sequences.max_body_length:
                text = pos.body[(-1 * sequences.max_body_length) :]
                text = text[text.find(" ") :]

            prediction = self._predict(text)

            end_time = time.time()
            pos.time = int(1000 * (end_time - start_time))

            if len(prediction) >= sequences.min_pred_length and prediction.score() >= sequences.min_score:
                pos.prediction = prediction
            else:
                pos.prediction = None

            if output_file is not None and ((i + 1) % sequences.save_step == 0 or i == (len(sequences) - 1)):
                sequences.save(output_file)

    def score(
        self,
        sequences: List[TextPredictionSequence],
        min_scores: Union[float, List[float]],
        expected_match_rate: Optional[float] = None,
    ) -> None:
        """Score a set of sequences (that have already been predicted).

        Args:
            sequences: List of sequences to score.
            min_scores: Minimum score to consider.
            expected_match_rate: Expected match rate to compute the minimum score.

        """

        if isinstance(min_scores, (float, int)):
            min_scores = [float(min_scores)]
        min_scores.sort()

        if expected_match_rate is not None and expected_match_rate >= 0 and expected_match_rate <= 1.0:
            min_scores.append(None)

        predictions = sequences.get_predictions()
        score_summary = []

        for min_score in min_scores:
            if min_score is None:
                match_rate = [summ["MatchRate"] for summ in score_summary]
                score = [summ["Score"] for summ in score_summary]

                if len(match_rate) < 2:
                    continue

                f = interp1d(
                    match_rate,
                    score,
                    bounds_error=False,
                    kind="linear",
                    fill_value="extrapolate",
                )
                min_score = float(f(expected_match_rate))

            triggered_preds = sequences.calculate_triggered_predictions(min_score, predictions)
            sequences.triggered_preds = triggered_preds

            scored_preds = sequences.score_triggered_predictions(
                triggered_preds, self.model, self.tokenizer, min_score=min_score
            )
            score_summary.append(scored_preds)

        sequences.score_summary = score_summary
