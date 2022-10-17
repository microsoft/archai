# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Text Predict classes and methods: prediction, sequence of predictions and predictor.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import ftfy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from archai.common import utils
from archai.nlp.metrics.text_predict.text_predict_model import (TextPredictONNXModel,
                                                                TextPredictTorchModel)
from archai.nlp.metrics.text_predict.text_predict_tokenizer import (TOKENIZER_WORD_TOKEN_SEPARATOR_SET,
                                                                    TextPredictTokenizer)


class Prediction:
    """Represents a single prediction.

    """

    # Constants for P(Accept) calculation
    a1 = 0.04218
    b1 = -0.1933

    # Penalties for calculating upper case score
    UPPER_PROB_PENALTY = 0.1
    UPPER_SCORE_PENALTY = 0.5

    def __init__(self,
                 text: str,
                 probability: float,
                 predictor: Optional[Predictor] = None,
                 input_ids: Optional[Tuple[int, ...]] = None,
                 token_ids: Optional[Tuple[int, ...]] = None,
                 complete: Optional[bool] = None,
                 match: Optional[bool] = None, 
                 score: Optional[float] = None):
        self.text = text
        self.probability = probability

        self.predictor = predictor

        self.input_ids = input_ids
        self.token_ids = token_ids

        self.complete = complete
        self.match = match
        self._score = score

    @classmethod
    def empty(cls, predictor: Optional[Predictor] = None) -> Prediction:
        return Prediction('', 0.0, predictor=predictor, complete=False)

    @classmethod
    def next_prediction(cls: Prediction,
                        prediction: Prediction,
                        next_text: str,
                        next_probability: float,
                        next_token_id: int) -> Prediction:
        
        next_prediction = Prediction(prediction.text + next_text,
                                     prediction.probability * next_probability,
                                     predictor=prediction.predictor,
                                     input_ids=prediction.input_ids,
                                     token_ids=prediction.token_ids + (next_token_id,))
        
        return next_prediction

    def show(self) -> str:
        return self.text.rstrip()

    def __len__(self) -> int:
        return len(self.show())

    def __str__(self) -> str:
        return self.show()

    def __repr__(self) -> str:
        return f'({self.text}, {self.probability:.5f}, {self.score():.3f})'

    def p_match(self) -> float:
        return self.probability

    def p_accept(self) -> float:
        result = self.probability * self.p_accept_given_match()

        if result < 0:
            return 0.0

        return result

    def p_accept_given_match(self) -> float:
        result = self.a1 * len(self) + self.b1

        if result < 0:
            return 0.0

        return result

    def score(self) -> float:
        if not self._score is None:
            return self._score

        a1 = 0.0
        b1 = 1.0

        length = len(self)
        score = self.probability * length * (a1 * length + b1)

        if self.predictor is None:
            if not self.is_empty():
                logging.info('Prediction environment not available. Ignoring additional parameters needed for score evaluation.')

            return score

        upper_token_ids = self.predictor.tp_tokenizer.upper_tokens.intersection(self.token_ids)

        if len(upper_token_ids) > 0:
            score = (self.probability - Prediction.UPPER_PROB_PENALTY) * length * (a1 * length + b1) - Prediction.UPPER_SCORE_PENALTY

        return score

    def char_accepted(self) -> float:
        return len(self) * self.p_accept()

    def all_ids(self) -> Tuple[int, ...]:
        if self.input_ids is None or self.token_ids is None:
            raise ValueError(f'Unable to determine combined ids for `{self}`.')

        return self.input_ids + self.token_ids

    def length_type(self) -> str:
        pred_len = len(self)

        if pred_len < 6:
            return '0:XS'

        if pred_len < 11:
            return '1:S'

        if pred_len < 16:
            return '2:M'

        return '3:L'

    def word_count(self) -> int:
        if len(self) == 0:
            return 0

        return len(re.findall(r'\s+', self.text.strip())) + 1

    def update_complete(self) -> bool:
        if self.input_ids is None or self.token_ids is None or self.predictor is None:
            raise ValueError(f'Unable to determine if `{self}` ends with a complete word.')

        self.complete = self.predictor.is_complete_word(tuple(self.input_ids + self.token_ids))

        return self.complete

    def is_valid(self) -> bool:
        if self.predictor is None:
            raise ValueError('Prediction environment not available; Not able to determine if prediction is valid.')

        if self.token_ids is None or len(self.token_ids) == 0:
            return False

        if len(self.show()) < self.predictor.MIN_PRED_LEN:
            return False

        if set(self.token_ids) & self.predictor.tp_tokenizer.INVALID_TOKENS:
            return False

        if self.complete is None:
            self.update_complete()

        if not self.complete:
            return False

        return True

    def is_empty(self) -> bool:
        return len(self.text) == 0

    def to_odict(self) -> OrderedDict:
        return OrderedDict({'Text': self.show(),
                            'Probability': self.probability,
                            'Length': len(self),
                            'Complete': self.complete,
                            'Match': self.match,
                            'PAccept': self.p_accept(),
                            'Score': self.score(),
                            'CharAccepted': self.char_accepted(),
                            'WordCount': self.word_count(),
                            'Tokens': self.token_ids})


@dataclass
class TextPredictionPosition:
    """Represents a single position inside the Text Prediction pipeline.

    """

    line_id: int
    char_id: int
    body: str
    body_continued: str
    prediction: Prediction | None = None
    time: int | None = None

    @classmethod
    def from_smart_compose_ljson(cls: TextPredictionPosition,
                                 line: str) -> TextPredictionPosition:
        d = json.loads(line)

        unique_ids = d['UniqueId'].split('-')
        if len(unique_ids) > 2:
            raise ValueError('Unable to split UniqueIds `{unique_ids}` into LineId and CharId.')

        line_id = int(unique_ids[0])
        char_id = int(unique_ids[1])

        if len(d.get('Suggestions', [])) > 0:
            suggestion = d['Suggestions'][0]
            text = suggestion['Suggestion']

            prob_keys = ['PMatch', 'Prob', 'Probability', 'Normalized Probability Score']
            probability = next((float(suggestion[k]) for k in prob_keys if k in suggestion.keys()), None)
            if probability is None:
                probability = 0.5

            prediction_score = float(suggestion.get('Score', probability))
            prediction = Prediction(text, probability, score=prediction_score)

        else:
            prediction = None

        return TextPredictionPosition(line_id=line_id,
                                      char_id=char_id,
                                      body=d['Body'],
                                      body_continued=d['BodyContinued'],
                                      prediction=prediction,
                                      time=d.get('Time', None))

    @property
    def unique_id(self) -> str:
        return f'{self.line_id}-{self.char_id}'

    def to_smart_compose_ljson(self) -> str:
        result = OrderedDict({'UniqueId': self.unique_id,
                              'Body': self.body,
                              'BodyContinued': self.body_continued})

        if not self.time is None:
            result['Time'] = self.time

        if self.prediction is None:
            result['Suggestions'] = []
        else:
            result['Suggestions'] = [{'Suggestion': self.prediction.show(),
                                      'Probability':  self.prediction.probability,
                                      'Score':  self.prediction.score()}]

        return json.dumps(result)


class TextPredictionSequence(OrderedDict):
    """Represents a sequence of positions inside the Text Prediction pipeline.

    """

    def __init__(self,
                 predictor: Optional[Predictor] = None,
                 max_body_len: Optional[int] = 1000000,
                 min_pred_len: Optional[int] = 6,
                 min_score: Optional[float] = 1.0,
                 save_step: Optional[int] = 100000,
                 current_paragraph_only: Optional[bool] = False):
        super().__init__()

        self.predictor = predictor

        self._perplexity = None
        self._word_count = None
        self._score_summary_list = None
        self._triggered_df = None

        self.max_body_len = max_body_len
        self.min_pred_len = min_pred_len
        self.min_score = min_score

        self.save_step = save_step
        self.current_paragraph_only = current_paragraph_only

    @classmethod
    def from_file(cls: TextPredictionSequence,
                  file_name: str,
                  file_type: str,
                  predictor: Optional[Predictor] = None,
                 **kwargs) -> TextPredictionSequence:
        if file_type == 'smartcompose':
            return TextPredictionSequence.from_smart_compose_file(file_name, predictor, **kwargs)

        if file_type == 'text':
            return TextPredictionSequence.from_text_file(file_name, predictor=predictor, **kwargs)

        return NotImplementedError

    @classmethod
    def from_smart_compose_file(cls: TextPredictionSequence,
                                file_name: str,
                                predictor: Optional[Predictor] = None,
                                **kwargs) -> TextPredictionSequence:
        logging.info(f'Loading SmartCompose file: {file_name}.')

        lines = []
        with open(file_name, encoding='utf-8') as f:
            lines = f.readlines()

        sequence = TextPredictionSequence(predictor, **kwargs)

        for line in lines:
            if len(line) < 10:
                logging.warning(f'Skipping line: `{line}`.')
                continue

            position = TextPredictionPosition.from_smart_compose_ljson(line)
            sequence[position.unique_id] = position

        return sequence

    @classmethod
    def from_text_file(cls: TextPredictionSequence,
                       file_name: str,
                       new_document_re: Optional[str] = '\\n\\n+',
                       predictor: Optional[Predictor] = None,
                       **kwargs) -> TextPredictionSequence:
        logging.info(f'Loading text file: {file_name}.')

        with open(file_name, encoding='utf-8') as f:
            text = f.read()

        lines = re.split(new_document_re, text, flags=re.DOTALL | re.MULTILINE)
        seq = TextPredictionSequence(predictor, **kwargs)

        for line_id, line in enumerate(lines):
            line = line.strip()
            line = ftfy.fix_text(line)

            if len(line) < 2:
                logging.warning(f'Skipping line `{line}` with line_id `{line_id}`.')

            for char_id in range(len(line)):
                position = TextPredictionPosition(line_id=line_id,
                                                  char_id=char_id,
                                                  body=line[:char_id],
                                                  body_continued=line[char_id:],
                                                  prediction=None,
                                                  time=None)
                seq[position.unique_id] = position

        return seq

    def save(self, output_filepath: str) -> None:
        output = [pos.to_smart_compose_ljson() + "\n" for pos in tqdm(self.values(), desc='Saving')]

        with open(output_filepath, 'w') as f:
            f.writelines(output)

    def save_score_summary(self, summary_file: str) -> None:
        if self._score_summary_list is None:
            logging.warning('Scores not calculated yet.')
            return

        with open(summary_file, 'w') as f:
            json_str = json.dumps(self._score_summary_list, indent=1)
            f.write(json_str)

    def save_settings(self, settings_file: str) -> None:
        with open(settings_file, 'w') as f:
            json_str = json.dumps(self.settings(), indent=1)
            f.write(json_str)

    def save_all(self,
                 output_dir: str,
                 predict_file: Optional[str] = 'output.ljson',
                 summary_file: Optional[str] = 'summary.json',
                 settings_file: Optional[str] = 'settings.json',
                 triggered_file: Optional[str] = 'triggered.csv') -> None:
        os.makedirs(output_dir, exist_ok=True)

        if predict_file is not None:
            predict_file_path = os.path.join(output_dir, predict_file)
            self.save(predict_file_path)

        if summary_file is not None:
            summary_file_path = os.path.join(output_dir, summary_file)
            logging.info(f'Saving summary: `{summary_file_path}`.')
            self.save_score_summary(summary_file_path)

        if settings_file is not None:
            settings_file_path = os.path.join(output_dir, settings_file)
            logging.info(f'Saving settings: `{settings_file_path}`.')
            self.save_settings(settings_file_path)

        if triggered_file is not None:
            if self._triggered_df is not None:
                triggered_file_path = os.path.join(output_dir, triggered_file)
                logging.info(f'Saving triggered information: `{triggered_file_path}`.')
                self._triggered_df.to_csv(index=False)
            else:
                logging.info('`triggered_df` not defined.')

    def settings(self) -> dict:
        settings = utils.attr_to_dict(self)
        settings['sequence_len'] = len(self)

        return settings

    def filter_keys_char_id(self, char_id: Optional[int] = None) -> List[str]:
        return [k for k, v in self.items() if v.char_id == char_id]

    def predict(self, output_filepath: Optional[str] = None) -> None:
        if self.predictor is None:
            raise ValueError('Predictor must be defined.')

        for i, pos in enumerate(tqdm(self.values(), desc='Predicting')):
            start_time = time.time()
            text = pos.body

            if self.current_paragraph_only:
                text = re.sub('^(.*\n)', '', text, flags=re.M)

            if len(text) > self.max_body_len:
                text = pos.body[(-1 * self.max_body_len):]
                text = text[text.find(' '):]

            prediction = self.predictor.predict(text)
            end_time = time.time()
            pos.time = int(1000 * (end_time - start_time))

            if len(prediction) >= self.min_pred_len and prediction.score() >= self.min_score:
                pos.prediction = prediction
            else:
                pos.prediction = None

            if output_filepath is not None and ((i + 1) % self.save_step == 0 or i == (len(self) - 1)):
                self.save(output_filepath)

    @property
    def perplexity(self) -> float:
        if hasattr(self, '_perplexity') and self._perplexity is not None:
            return self._perplexity

        return self.get_perplexity()

    def get_perplexity(self) -> float:
        if self.predictor is None:
            logging.warning('Perplexity can not be calculated with unknown predictor.')
            
            return None

        loss_sum = 0.0
        token_ids_len_sum = 0

        for unique_id in self.filter_keys_char_id(1):
            text = self[unique_id].body + self[unique_id].body_continued
            text = self.predictor.tp_tokenizer.clean(text)
            token_ids = self.predictor.tp_tokenizer.encode(text)

            token_ids_len_sum += len(token_ids)
            loss_sum += len(token_ids) * self.predictor.tp_model.get_loss(tuple(token_ids))

        perplexity = np.exp(loss_sum/token_ids_len_sum)
        self._perplexity = perplexity

        return perplexity

    @property
    def word_count(self) -> int:
        if hasattr(self, '_word_count') and self._word_count is not None:
            return self._word_count

        return self.get_word_count()

    def get_word_count(self) -> int:
        word_count = 0

        for unique_id in self.filter_keys_char_id(1):
            text = self[unique_id].body + self[unique_id].body_continued
            word_count += len(re.findall(r'[^\s]+', text, re.DOTALL | re.MULTILINE))

        self._word_count = word_count

        return word_count

    def get_predictions(self) -> pd.Series:
        predictions_list = []

        for pos in self.values():
            prediction = pos.prediction

            if prediction is not None:
                body_continued = pos.body_continued[:len(prediction)]
                prediction.match = prediction.show() == body_continued

                min_len = min(len(prediction.show()), len(body_continued))
                last_match_char = next((i for i in range(min_len) if prediction.show()[i] != body_continued[i]), min_len)
                length_type = prediction.length_type()
                
                prediction_odict = prediction.to_odict()
                p_accept_given_match = prediction.p_accept_given_match()

            else:
                body_continued, min_len, last_match_char = True, 0, 0
                
                prediction_odict = OrderedDict([('Text', ''), ('Probability', 0.0), ('Length', 0), ('Complete', False), ('Match', None), ('PAccept', 0.0), ('Score', 0.0), ('CharAccepted', 0.0), ('WordCount', 0), ('Tokens', None)])
                length_type = ''
                p_accept_given_match = 0.0

            prediction_odict['Line'] = pos.line_id
            prediction_odict['Char'] = pos.char_id
            prediction_odict['BodyContinued'] = body_continued
            prediction_odict['Type'] = length_type
            prediction_odict['LastMatchChar'] = last_match_char
            prediction_odict['NextTrigger'] = pos.char_id + last_match_char + 1
            prediction_odict['PAcceptGivenMatch'] = p_accept_given_match

            predictions_list.append(prediction_odict)

        predictions_df = pd.DataFrame(predictions_list)
        predictions_df_columns = ['Line', 'Char', 'Text', 'BodyContinued']
        predictions_df_columns = predictions_df_columns + predictions_df.columns.drop(predictions_df_columns + ['Complete', 'Tokens']).tolist()
        predictions_df = predictions_df[predictions_df_columns]

        return predictions_df

    def calc_triggered_predictions(self,
                                   min_score: float,
                                   predictions_df: Optional[pd.Series] = None) -> pd.DataFrame:
        if predictions_df is None:
            predictions_df = self.get_predictions()

        triggered_list = []

        # Keep count on where it triggered
        curr_line = -1
        curr_char = -1

        # Triggered is an array that denotes if suggestion is regarded as 'triggered':
        # -1 - the score was too low
        #  0 - score OK, but something is being shown and current suggestion could not be shown
        #  1 - suggestion shown
        triggered = np.full((len(predictions_df.index),), -1)

        score_values = predictions_df['Score'].values

        for i in tqdm(range(predictions_df.shape[0]), desc=f'Scoring with {min_score:.2f}'):
            if score_values[i] < min_score:
                continue

            if predictions_df['Line'][i] < curr_line:
                msg = f'Incorrect order of lines in the file (current line = {curr_line}, '
                msg += f'processed line = {predictions_df["Line"][i]}; current char = {curr_char}, '
                msg += f'processed char = {predictions_df["Char"][i]}'
                
                raise ValueError(msg)

            triggered[i] = 0

            if predictions_df['Line'][i] > curr_line or predictions_df['Char'][i] > curr_char:
                d = predictions_df.iloc[i].to_dict()
                
                curr_line = d['Line']
                curr_char = d['Char'] + d['LastMatchChar']
                
                triggered_list.append(d)
                triggered[i] = 1

        triggered_df = pd.DataFrame(triggered_list)
        trigger_column_name = f'Trigger:{min_score}'
        predictions_df[trigger_column_name] = triggered

        if len(triggered_df.index) == 0:
            logging.warning(f'No suggestions found for min_score: {min_score}.')

        return triggered_df

    def score_triggered_predictions(self,
                                    triggered_df: pd.DataFrame,
                                    min_score: Optional[float] = None) -> OrderedDict:
        summary = OrderedDict()

        summary['Score'] = min_score
        summary['TotalEvalPoints'] = len(self)
        summary['TotalWordCount'] = self.word_count
        summary['Perplexity'] = self.perplexity

        summary['SuggestionsShown'] = len(triggered_df.index)
        summary['SuggestionsMatched'] = int(np.sum(triggered_df['Match'])) if len(triggered_df.columns) else 0
        summary['SuggestionsAccepted'] = int(np.sum(triggered_df['Match'] * triggered_df['PAcceptGivenMatch'])) if len(triggered_df.columns) else 0
        summary['SuggestionRatePerWord'] = summary['SuggestionsShown']/summary['TotalWordCount']
        summary['SuggestionRatePerChar'] = summary['SuggestionsShown']/summary['TotalEvalPoints']

        summary['MatchRate'] = np.mean(triggered_df['Match']) if len(triggered_df.columns) else 0
        summary['AcceptRate'] = np.mean(triggered_df['Match'] * triggered_df['PAcceptGivenMatch']) if len(triggered_df.columns) else 0
        summary['CharMatched'] = int(np.sum(triggered_df['Match'] * triggered_df['Length'])) if len(triggered_df.columns) else 0
        summary['CharAccepted'] = int(np.sum(triggered_df['Match'] * triggered_df['PAcceptGivenMatch'] * triggered_df['Length'])) if len(triggered_df.columns) else 0
        summary['CharMatchRate'] = summary['CharMatched']/summary['TotalEvalPoints']
        summary['CharAcceptRate'] = summary['CharAccepted']/summary['TotalEvalPoints']

        summary['SuggestionsShownByType'] = triggered_df.groupby(['Type']).size().to_dict() if len(triggered_df.columns) else None
        summary['SuggestionsMatchedByType'] = triggered_df[triggered_df['Match']].groupby(['Type']).size().to_dict() if len(triggered_df.columns) else 0
        summary['MatchRateByType'] = triggered_df.groupby(['Type']).agg({'Match':'mean'}).to_dict()['Match'] if len(triggered_df.columns) else None

        summary['SuggestionsShownByWordCount'] = triggered_df.groupby(['WordCount']).size().to_dict() if len(triggered_df.columns) else None
        summary['SuggestionsMatchedByWordCount'] = triggered_df[triggered_df['Match']].groupby(['WordCount']).size().to_dict() if len(triggered_df.columns) else None
        summary['MatchRateByWordCount'] = triggered_df.groupby(['WordCount']).agg({'Match':'mean'}).to_dict()['Match'] if len(triggered_df.columns) else None

        return summary

    def score(self,
              min_scores: Union[list, float],
              expected_match_rate: Optional[float] = None) -> List[Any]:
        if isinstance(min_scores, (float, int)):
            min_scores = [float(min_scores)]
            
        min_scores.sort()

        if expected_match_rate is not None and expected_match_rate >=0 and expected_match_rate <= 1.0:
            min_scores.append(None)

        predictions_df = self.get_predictions()
        summary_list = []

        for min_score in min_scores:
            if min_score is None:
                # Perform a fit of quadratic equation
                match_rates = [summ['MatchRate'] for summ in summary_list]
                scores = [summ['Score'] for summ in summary_list]

                if len(match_rates) < 2:
                    logging.warning(f'Not enough points to calculate score for the expected match rate: {expected_match_rate}.')
                    continue

                f = interp1d(match_rates, scores, bounds_error=False, kind='linear', fill_value='extrapolate')
                min_score = float(f(expected_match_rate))

                logging.debug(f'Expected_score: {min_score} at {expected_match_rate}.')

            triggered_df = self.calc_triggered_predictions(min_score, predictions_df)
            self._triggered_df = triggered_df

            summary_dict = self.score_triggered_predictions(triggered_df, min_score)
            summary_list.append(summary_dict)

        self._score_summary_list = summary_list

        return summary_list


class Predictor:
    """Runs the Text Predict pipeline.

    Usage:
        >>> model = TextPredictModel(model_path)
        >>> tokenizer = TextPredictTokenizer(vocab_path)
        >>> tp = Predictor(model, tokenizer)
        >>> seq = TextPredictionSequence.from_smart_compose_file('file.ljson', tp)
        >>> seq.predict()
        >>> score = seq.score([1, 1.5, 2, 2.5, 3, 3.5, 4, 5], expected_match_rate=0.5)
        >>> print(json.dumps(score, indent=2))
    
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

    # Begin-of-sentence token
    BOS_TOKEN_ID = None

    def __init__(self, tp_model, tp_tokenizer) -> None:
        self.tp_model = tp_model
        self.tp_tokenizer = tp_tokenizer
        self.bos_id = self.BOS_TOKEN_ID

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

    def settings(self) -> Dict[str, Any]:
        settings = utils.attr_to_dict(self)

        return settings

    @staticmethod
    def score(prob: float, length: int, a1: Optional[float] = 0.0, b1: Optional[float] = 1.0):
        return prob * (a1 * length + b1) * length

    @functools.lru_cache(maxsize=1024)
    def filter_next_tokens(self,
                           input_ids: Optional[Tuple[int, ...]] = (),
                           filter_prefix: Optional[str] ='') -> Tuple[int, ...]:
        next_token_probs = self.tp_model.get_probs(input_ids)
        filter_prefix_len = len(filter_prefix)

        if filter_prefix_len == 0:
            result = [((idx,), prob, len(self.tp_tokenizer[idx])) for idx, prob in enumerate(next_token_probs)]
        else:
            filter_next_token_ids = self.tp_tokenizer.filter_token_tuple_ids(filter_prefix)
            result = [(tuple_idx, next_token_probs[tuple_idx[0]], len(self.tp_tokenizer[tuple_idx[0]]) - filter_prefix_len) \
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

            pred_text = self.tp_tokenizer.decode(idxs)[len(prefix):]
            prediction = Prediction(pred_text, prob/prob_sum, predictor=self, input_ids=input_ids, token_ids=idxs)
            
            logging.debug('predict_word_prefix: prefix: `%s`: # calcs: %s; time: %.3f ms', prefix, calc_count, 1000*(time.time() - start_time))

        if debug:
            prediction.calc_count = calc_count
            prediction.filtered_list = filtered_list

        return prediction

    @functools.lru_cache(maxsize=1024)
    def is_complete_word(self, input_ids: Tuple[int, ...]) -> bool:
        if len(input_ids) > 0 and self.tp_tokenizer[input_ids[-1]][-1] in TOKENIZER_WORD_TOKEN_SEPARATOR_SET:
            return True

        probs = self.tp_model.get_probs(input_ids)
        prob_sum = sum([prob for idx, prob in enumerate(probs) if idx in self.tp_tokenizer.TOKENIZER_WORD_TOKEN_SEPARATOR])

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

        clean_trunc_text = self.tp_tokenizer.clean(trunc_text, add_bos_text=is_full_len)
        context, prefix = self.tp_tokenizer.find_context_prefix(clean_trunc_text)

        input_ids = tuple(self.tp_tokenizer.encode(context))

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

            next_token_id, next_prob = self.tp_model.get_top_token_prob(tuple(prediction.all_ids()))
            next_text = self.tp_tokenizer.decode([next_token_id])

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


def run_score(default_path: str,
              model_path: str,
              vocab_path: str,
              input_file_path: str,
              input_file_type: str,
              model_type: str,
              score_type: Optional[str] = 'torch',
              save_step: Optional[int] = 100000,
              min_score: Optional[float] = 1.0,
              max_score: Optional[float] = 5.0,
              score_step: Optional[float] = 0.1,
              expected_match_rate: Optional[float] = 0.5,
              current_paragraph_only: Optional[bool] = False,
              max_body_len: Optional[int] = 10000,
              max_seq_len: Optional[int] = 30,
              min_pred_len: Optional[int] = 6) -> None:
    # Defines the scoring output path
    output_path = utils.full_path(os.path.join(default_path, 'score'), create=True)

    # Loads the vocab/tokenizer from provided path
    tp_tokenizer = TextPredictTokenizer(vocab_path)
    space_token_id = tp_tokenizer.tokenizer.encode(' ')[0]

    # Torch-based model
    if score_type == 'torch':
        tp_model = TextPredictTorchModel(model_type, model_path, space_token_id, max_seq_len)

    # ONNX-based model
    elif score_type == 'onnx':
        tp_model = TextPredictONNXModel(model_type, model_path, space_token_id, max_seq_len)

    # Creates the Text Predict (Predictor) instance
    predictor = Predictor(tp_model, tp_tokenizer)
    predictor.MAX_INPUT_TEXT_LEN = max_body_len

    # Retrieves sequences that will be predicted
    seq = (TextPredictionSequence).from_file(input_file_path,
                                             input_file_type,
                                             predictor,
                                             save_step=save_step,
                                             min_score=min_score,
                                             current_paragraph_only=current_paragraph_only,
                                             min_pred_len=min_pred_len)
    seq.predict()

    # Calculates the scores
    min_scores = np.arange(min_score, max_score, score_step).tolist()
    seq.score(min_scores, expected_match_rate)
    seq.save_all(output_path)
