# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Entrypoint of prediction representations: single and sequence.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import ftfy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from archai.nlp.metrics.text_predict.predictor import TextPredictor
from archai.nlp.metrics.text_predict.text_predict_utils import get_settings


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
                 predictor: Optional[TextPredictor] = None,
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
    def empty(cls, predictor: TextPredictor) -> Prediction:
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

        upper_token_ids = self.predictor.vocab_wrapper.UPPER_TOKENS.intersection(self.token_ids)

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

        if set(self.token_ids) & self.predictor.vocab_wrapper.INVALID_TOKENS:
            return False

        if self.complete is None:
            self.update_complete()

        if not self.complete:
            return False

        return True

    def is_empty(self):
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
    """Represents a single position for text prediction.

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
    """Represents a sequence of text predictions.

    Usage:
        >>> model = create_model('transformers', 'distilgpt2', max_seq_len=1024)
        >>> tokenizer = create_tokenizer('transformers', 'gpt2')
        >>> tp = TextPredictor(model, tokenizer)
        >>> seq = TextPredictionSequence.from_smart_compose_file('GSuiteCompete10pc.ljson', tp)
        >>> seq.predict()
        >>> score = seq.score([1, 1.5, 2, 2.5, 3, 3.5, 4, 5], expected_match_rate = 0.5)
        >>> print(json.dumps(score, indent=2))

    """

    def __init__(self,
                 predictor: Optional[TextPredictor] = None,
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
                  predictor: Optional[TextPredictor] = None,
                 **kwargs) -> TextPredictionSequence:
        if file_type == 'smartcompose':
            return TextPredictionSequence.from_smart_compose_file(file_name, predictor, **kwargs)

        if file_type == 'text':
            return TextPredictionSequence.from_text_file(file_name, predictor=predictor, **kwargs)

        return NotImplementedError

    @classmethod
    def from_smart_compose_file(cls: TextPredictionSequence,
                                file_name: str,
                                predictor: Optional[TextPredictor] = None,
                                **kwargs) -> TextPredictionSequence:
        logging.info(f'Loading smartcompose file from {file_name}')

        lines = []
        with open(file_name) as f:
            lines = f.readlines()

        sequence = TextPredictionSequence(predictor, **kwargs)

        for line in lines:
            if len(line) < 10:
                logging.warning(f'Skipping line `{line}`')
                continue

            position = TextPredictionPosition.from_smart_compose_ljson(line)
            sequence[position.unique_id] = position

        return sequence

    @classmethod
    def from_text_file(cls: TextPredictionSequence,
                       file_name: str,
                       new_document_re: Optional[str] = '\\n\\n+',
                       predictor: Optional[TextPredictor] = None,
                       **kwargs) -> TextPredictionSequence:
        logging.info(f'Loading text file from {file_name}')

        with open(file_name, encoding='utf-8') as f:
            text = f.read()

        lines = re.split(new_document_re, text, flags=re.DOTALL | re.MULTILINE)
        seq = TextPredictionSequence(predictor, **kwargs)

        for line_id, line in enumerate(lines):
            line = line.strip()
            line = ftfy.fix_text(line)

            if len(line) < 2:
                logging.warning(f'Skipping line `{line}` with line_id `{line_id}`')

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
        output = [pos.to_smart_compose_ljson() + "\n" for pos in self.values()]

        with open(output_filepath, 'w') as f:
            f.writelines(output)

    def save_score_summary(self, summary_file: str) -> None:
        if self._score_summary_list is None:
            logging.warning('Scores not calculated yet - not saving them')
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
            logging.info(f'Saving scoring summary to `{summary_file_path}` file')

            summary_file_path = os.path.join(output_dir, summary_file)
            self.save_score_summary(summary_file_path)

        if settings_file is not None:
            logging.info(f'Saving settings info to `{settings_file_path}` file')

            settings_file_path = os.path.join(output_dir, settings_file)
            self.save_settings(settings_file_path)

        if triggered_file is not None:
            if self._triggered_df is not None:
                logging.info(f'Saving triggered info to `{triggered_file_path}` file')

                triggered_file_path = os.path.join(output_dir, triggered_file)
                self._triggered_df.to_csv(index=False)
            else:
                logging.info('triggered_df not defined - not saving')

    def settings(self) -> dict:
        settings = get_settings(self)
        settings['sequence_len'] = len(self)

        return settings

    def filter_keys_char_id(self, char_id: Optional[int] = None) -> List[str]:
        return [k for k, v in self.items() if v.char_id == char_id]

    def predict(self, output_filepath: Optional[str] = None) -> None:
        if self.predictor is None:
            raise ValueError('TextPredictor must be defined to run text prediction')

        for i, pos in enumerate(self.values()):
            start_time = time.time()
            text = pos.body

            if self.current_paragraph_only:
                text = re.sub('^(.*\n)', '', text, flags=re.M)

            if len(text) > self.max_body_len:
                text = pos.body[(-1 * self.max_body_len):]
                text = text[text.find(' '):]

            prediction = self.predictor.predict(text)
            end_time = time.time()
            pos.time = int(1000*(end_time - start_time))

            if len(prediction) >= self.min_pred_len and prediction.score() >= self.min_score:
                pos.prediction = prediction
            else:
                pos.prediction = None

            if output_filepath is not None and ((i+1) % self.save_step == 0 or i == (len(self) - 1)):
                self.save(output_filepath)

    @property
    def perplexity(self) -> float:
        if hasattr(self, '_perplexity') and self._perplexity is not None:
            return self._perplexity

        return self.get_perplexity()

    def get_perplexity(self) -> float:
        if self.predictor is None:
            logging.warning('TextPredictor not defined. Perplexity not calculated')
            
            return None

        loss_sum = 0.0
        token_ids_len_sum = 0

        for unique_id in self.filter_keys_char_id(1):
            text = self[unique_id].body + self[unique_id].body_continued
            text = self.predictor.vocab_wrapper.clean(text)
            token_ids = self.predictor.vocab_wrapper.encode(text)

            token_ids_len_sum += len(token_ids)
            loss_sum += len(token_ids) * self.predictor.model_wrapper.get_loss(tuple(token_ids))

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
        curr_line = -1 # Keep count on where it triggered
        curr_char = -1

        # Triggered is an array that denotes if suggestion is regarded as 'triggered':
        # -1 - the score was too low
        #  0 - score OK, but something is being shown and current suggestion could not be shown
        #  1 - suggestion shown
        triggered = np.full((len(predictions_df.index),), -1)

        score_values = predictions_df['Score'].values

        for i in range(predictions_df.shape[0]):
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
            logging.warning('No suggestions found for min_score %s', min_score)

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
                    logging.warning('Not enough points to calculate score for the expected match rate of %.3f', expected_match_rate)
                    continue

                f = interp1d(match_rates, scores, bounds_error=False, kind='linear', fill_value='extrapolate')
                min_score = float(f(expected_match_rate))

                logging.debug('Expected_score = %s at %s', min_score, expected_match_rate)

            triggered_df = self.calc_triggered_predictions(min_score, predictions_df)
            self._triggered_df = triggered_df

            summary_dict = self.score_triggered_predictions(triggered_df, min_score)
            summary_list.append(summary_dict)

        self._score_summary_list = summary_list

        return summary_list
