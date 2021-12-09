# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Simulates user experience when using Text Prediction.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional

import ftfy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from archai.nlp.scoring_metrics.scoring_utils import get_settings
from archai.nlp.scoring_metrics.text_predictor import Prediction, TextPredictor


@dataclass
class TextPredictionPosition:
    """Represents single position for text prediction.
    """
    line_id: int
    char_id: int
    body: str
    body_continued: str
    prediction: Prediction|None = None
    time: int|None = None

    @classmethod
    def from_smart_compose_ljson(cls, line):
        """Factory method to create TextPredictionPosition from SmartCompose ljson line.

        Args:
            line ([type]): SmartCompose ljson line

        Raises:
            ValueError: If it cannot parse the line

        Returns:
            [TextPredictionPosition]: with parameters obtained from the ljson line
        """
        d = json.loads(line)
        unique_ids = d["UniqueId"].split('-')
        if len(unique_ids) > 2:
            raise ValueError("Unable to split UniqueIds '{unique_ids}' into LineId and CharId")
        line_id = int(unique_ids[0])
        char_id = int(unique_ids[1])
        if len(d.get("Suggestions", [])) > 0:
            suggestion = d["Suggestions"][0]
            text = suggestion["Suggestion"]
            prob_keys = ["PMatch", "Prob", "Probability", "Normalized Probability Score"]
            probability = next((float(suggestion[k]) for k in prob_keys if k in suggestion.keys()), None)
            if probability is None:
                probability = 0.5

            prediction_score = float(suggestion.get("Score", probability))

            prediction = Prediction(text, probability, score=prediction_score)
        else:
            prediction = None

        return TextPredictionPosition(line_id=line_id,
                                      char_id=char_id,
                                      body=d["Body"],
                                      body_continued=d["BodyContinued"],
                                      prediction=prediction,
                                      time=d.get("Time", None)
                                      )

    @property
    def unique_id(self) -> str:
        return f"{self.line_id}-{self.char_id}"

    def to_smart_compose_ljson(self) -> str:
        result:OrderedDict[str,Any] = OrderedDict({
            "UniqueId": self.unique_id,
            "Body": self.body,
            "BodyContinued": self.body_continued})

        if not self.time is None:
            result["Time"] = self.time

        if self.prediction is None:
            result["Suggestions"] = []
        else:
            result["Suggestions"] = [{
                "Suggestion": self.prediction.show(),
                "Probability":  self.prediction.probability,
                "Score":  self.prediction.score()
            }]

        return json.dumps(result)


class TextPredictionSequence(OrderedDict):
    """Represents sequence of text prediction to simulate user experience of predicting text.

    To use it:
model = create_model("transformers", "distilgpt2", max_seq_len=1024)
tokenizer = create_tokenizer("transformers", "gpt2")
tp = TextPredictor(model, tokenizer)
seq = TextPredictionSequence.from_smart_compose_file("GSuiteCompete10pc.ljson", tp)
seq.predict()
score = seq.score([1, 1.5, 2, 2.5, 3, 3.5, 4, 5], expected_match_rate = 0.5)
print(json.dumps(score, indent=2))
    """
    def __init__(self, predictor:Optional[TextPredictor] = None,
                 max_body_len = 1_000_000, min_pred_len=6, min_score=1.0,
                 save_step=100_000, current_paragraph_only=False):
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
    def from_file(cls, file_name: str, file_type: str, predictor:Optional[TextPredictor] = None,
                 **kwargs):
        # TODO:Fixme - put more file formats
        if file_type == "smartcompose":
            return TextPredictionSequence.from_smart_compose_file(file_name, predictor,
                **kwargs)
        if file_type == "text":
            return TextPredictionSequence.from_text_file(file_name, predictor=predictor,
                **kwargs)

        return NotImplementedError

    @classmethod
    def from_smart_compose_file(cls, file_name: str,
                predictor:Optional[TextPredictor] = None, **kwargs) -> TextPredictionSequence:
        """Load SmartCompose .json file format.
        """
        logging.info(f"Loading smartcompose file from {file_name}")
        lines = []
        with open(file_name) as f:
            lines = f.readlines()

        sequence = TextPredictionSequence(predictor, **kwargs)
        for line in lines:
            if len(line) < 10:
                logging.warning(f"Skipping line '{line}'")
                continue
            position = TextPredictionPosition.from_smart_compose_ljson(line)
            sequence[position.unique_id] = position

        return sequence

    @classmethod
    def from_text_file(cls, file_name: str, new_document_re: str = "\\n\\n+",
                       predictor:Optional[TextPredictor] = None, **kwargs) -> TextPredictionSequence:
        """Load text file and convert it to TextPredictionSequence object.

        Args:
            file_name (str): file to load the text from
            new_document_re (str, optional): Regular expression that splits the file into separate documents in TextPredictionSequence
            (Defaults to "\n\n+")
            predictor (TextPredictionSequence, optional): [description]. Defaults to None.

        Returns:
            TextPredictionSequence: [description]
        """
        logging.info(f"Loading text file from {file_name}")
        with open(file_name, encoding="utf-8") as f:
            text = f.read()
        lines = re.split(new_document_re, text, flags=re.DOTALL | re.MULTILINE)

        seq = TextPredictionSequence(predictor, **kwargs)
        for line_id, line in enumerate(lines):
            line = line.strip()
            line = ftfy.fix_text(line)
            if len(line) < 2:
                logging.warning(f"Skipping line '{line}' with line_id '{line_id}'")
            for char_id in range(len(line)):
                position = TextPredictionPosition(
                    line_id=line_id,
                    char_id=char_id,
                    body=line[:char_id],
                    body_continued=line[char_id:],
                    prediction=None,
                    time=None)

                seq[position.unique_id] = position

        return seq

    def save(self, output_filepath: str) -> None:
        """Save SmartCompose file to output_filepath.
        """
        output = [pos.to_smart_compose_ljson() + "\n" for pos in self.values()]
        with open(output_filepath, 'w') as f:
            f.writelines(output)

    def save_score_summary(self, summary_file) -> None:
        if self._score_summary_list is None:
            logging.warning("Scores not calculated yet - not saving them")
            return

        with open(summary_file, 'w') as f:
            json_str = json.dumps(self._score_summary_list, indent=1)
            f.write(json_str)

    def save_settings(self, settings_file) -> None:
        with open(settings_file, 'w') as f:
            json_str = json.dumps(self.settings(), indent=1)
            f.write(json_str)

    def save_all(self, output_dir: str, predict_file="Output.ljson", summary_file="summary.json", settings_file="settings.json", triggered_file="triggered.csv") -> None:
        """Save result, scoring and settings into specified directory

        Args:
            output_dir (str): Directory to save results to
            predict_file (str|None): Smartcompose file with all the predictions
            summary_file (str|None): File with scoring summary
            settings_file (str|None): File with all the settings
        """
        os.makedirs(output_dir, exist_ok=True)
        if predict_file is not None:
            predict_file_path = os.path.join(output_dir, predict_file)
            self.save(predict_file_path)

        if summary_file is not None:
            summary_file_path = os.path.join(output_dir, summary_file)
            logging.info(f"Saving scoring summary to '{summary_file_path}' file")
            self.save_score_summary(summary_file_path)

        if settings_file is not None:
            settings_file_path = os.path.join(output_dir, settings_file)
            logging.info(f"Saving settings info to '{settings_file_path}' file")
            self.save_settings(settings_file_path)

        if triggered_file is not None:
            if self._triggered_df is not None:
                triggered_file_path = os.path.join(output_dir, triggered_file)
                logging.info(f"Saving triggered info to '{triggered_file_path}' file")
                self._triggered_df.to_csv(index=False)
            else:
                logging.info("triggered_df not defined - not saving")


    def settings(self) -> dict:
        """Return settable parameters of this object.
        """
        settings = get_settings(self)
        settings["sequence_len"] = len(self)
        return settings

    def filter_keys_char_id(self, char_id: int = None) -> List[str]:
        """Filter the keys of the TextPredictionSequence based on char_id

        Returns:
            list(str): list of keys that have given char_id
        """
        keys = [k for k, v in self.items() if v.char_id == char_id]
        return keys

    def predict(self, output_filepath: str = None) -> None:
        """Run the TextPredictor at every point and record the observed output

        Args:
            output_filepath (str, optional): Save the file at SAVE_STEP intervals,
            if given.
        """
        if self.predictor is None:
            raise ValueError("TextPredictor must be defined to run text prediction")

        for i, pos in enumerate(self.values()):
            start_time = time.time()
            text = pos.body
            if self.current_paragraph_only:
                text = re.sub("^(.*\n)", "", text, flags=re.M)

            if len(text) > self.max_body_len:
                text = pos.body[(-1*self.max_body_len):] # Truncate
                text = text[text.find(' '):]             # Remove partial token

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
    def perplexity(self):
        """Perplexity property, so that you can access it as:
        obj.perplexity

        It will evaluate it only if it was not calculated beforehand.
        """
        if hasattr(self, '_perplexity') and self._perplexity is not None:
            return self._perplexity
        return self.get_perplexity()

    def get_perplexity(self):
        """Returns perplexity of the predicted text

        Returns:
            float: perplexity
        """
        if self.predictor is None:
            logging.warning("TextPredictor not defined. Perplexity not calculated")
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
    def word_count(self):
        if hasattr(self, '_word_count') and self._word_count is not None:
            return self._word_count
        return self.get_word_count()

    def get_word_count(self) -> int:
        """Returns:
            int: word count of the TextPredictionSequence,
            defined as # of continuous non-space characters
        """
        word_count = 0
        for unique_id in self.filter_keys_char_id(1):
            text = self[unique_id].body + self[unique_id].body_continued
            word_count += len(re.findall(r'[^\s]+', text, re.DOTALL | re.MULTILINE))

        self._word_count = word_count
        return word_count

    def get_predictions(self) -> pd.Series:
        """Extract predictions from the TextPredictionSequence and return them in DataFrame format.

        Returns:
            pd.DataFrame: predictions
        """
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
                prediction_odict = OrderedDict(
                    [('Text', ''), ('Probability', 0.0), ('Length', 0), ('Complete', False), ('Match', None), ('PAccept', 0.0), ('Score', 0.0), ('CharAccepted', 0.0), ('WordCount', 0), ('Tokens', None)]
                    )
                length_type = ''
                p_accept_given_match = 0.0

            prediction_odict["Line"] = pos.line_id
            prediction_odict["Char"] = pos.char_id
            prediction_odict["BodyContinued"] = body_continued
            prediction_odict["Type"] = length_type
            prediction_odict["LastMatchChar"] = last_match_char
            prediction_odict["NextTrigger"] = pos.char_id + last_match_char + 1
            prediction_odict["PAcceptGivenMatch"] = p_accept_given_match
            predictions_list.append(prediction_odict)

        predictions_df = pd.DataFrame(predictions_list)

        predictions_df_columns = ["Line", "Char", "Text", "BodyContinued"]
        predictions_df_columns = predictions_df_columns + predictions_df.columns.drop(predictions_df_columns + ["Complete", "Tokens"]).tolist()
        predictions_df = predictions_df[predictions_df_columns]

        return predictions_df

    def calc_triggered_predictions(self, min_score: float, predictions_df:Optional[pd.Series] = None) -> pd.DataFrame:
        """Simulate user experience and calculate which predictions were actually triggered.

        Args:
            min_score (float): min_score that results in showing the predictions
            predictions_df (pd.DataFrame, optional): DataFrame with all the predictions.
            It is recalculated internally if not given.

        Returns:
            pd.DataFrame: DataFrame with triggered predictions
        """
        if predictions_df is None:
            predictions_df = self.get_predictions()
        triggered_list = []

        curr_line = -1 # Keep count on where it triggered
        curr_char = -1
        # Triggered is an array that denotes if suggestion is regarded as "triggered":
        # -1 - the score was too low
        #  0 - score OK, but something is being shown and current suggestion could not be shown
        #  1 - suggestion shown
        triggered = np.full((len(predictions_df.index),), -1)
        score_values = predictions_df["Score"].values
        for i in range(predictions_df.shape[0]):
            if score_values[i] < min_score:
                continue
            if predictions_df["Line"][i] < curr_line:
                msg = f"Incorrect order of lines in the file (current line = {curr_line}, "
                msg += f"processed line = {predictions_df['Line'][i]}; current char = {curr_char}, "
                msg += f"processed char = {predictions_df['Char'][i]}"
                raise ValueError(msg)

            triggered[i] = 0
            if predictions_df["Line"][i] > curr_line or predictions_df["Char"][i] > curr_char:
                d = predictions_df.iloc[i].to_dict()
                curr_line = d["Line"]
                curr_char = d["Char"] + d["LastMatchChar"]
                triggered_list.append(d)
                triggered[i] = 1

        triggered_df = pd.DataFrame(triggered_list)

        trigger_column_name = f"Trigger:{min_score}"
        predictions_df[trigger_column_name] = triggered

        if len(triggered_df.index) == 0:
            logging.warning("No suggestions found for min_score %s", min_score)

        return triggered_df

    def score_triggered_predictions(self, triggered_df: pd.DataFrame, min_score: float = None) -> OrderedDict:
        """It calculates statistics/scores for triggered predictions.

        Args:
            triggered_df (pd.DataFrame): DataFrame with triggered predictions
            min_score (float, optional): Minimum score, only used in output summary.

        Returns:
            OrderedDict: [description]
        """
        summary = OrderedDict()
        summary["Score"] = min_score
        summary["TotalEvalPoints"] = len(self)
        summary["TotalWordCount"] = self.word_count
        summary["Perplexity"] = self.perplexity
        summary["SuggestionsShown"] = len(triggered_df.index)
        summary["SuggestionsMatched"] = int(np.sum(triggered_df["Match"])) \
            if len(triggered_df.columns) else 0
        summary["SuggestionsAccepted"] = int(np.sum(triggered_df["Match"] * triggered_df["PAcceptGivenMatch"])) \
            if len(triggered_df.columns) else 0
        summary["SuggestionRatePerWord"] = summary["SuggestionsShown"]/summary["TotalWordCount"]
        summary["SuggestionRatePerChar"] = summary["SuggestionsShown"]/summary["TotalEvalPoints"]
        summary["MatchRate"] = np.mean(triggered_df["Match"]) \
            if len(triggered_df.columns) else 0
        summary["AcceptRate"] = np.mean(triggered_df["Match"] * triggered_df["PAcceptGivenMatch"]) \
            if len(triggered_df.columns) else 0
        summary["CharMatched"] = int(np.sum(triggered_df["Match"] * triggered_df["Length"])) \
            if len(triggered_df.columns) else 0
        summary["CharAccepted"] = int(np.sum(triggered_df["Match"] * triggered_df["PAcceptGivenMatch"] * triggered_df["Length"])) \
            if len(triggered_df.columns) else 0
        summary["CharMatchRate"] = summary["CharMatched"]/summary["TotalEvalPoints"]
        summary["CharAcceptRate"] = summary["CharAccepted"]/summary["TotalEvalPoints"]
        summary["SuggestionsShownByType"] = triggered_df.groupby(["Type"]).size().to_dict() \
            if len(triggered_df.columns) else None
        summary["SuggestionsMatchedByType"] = triggered_df[triggered_df["Match"]].groupby(["Type"]).size().to_dict() \
            if len(triggered_df.columns) else 0
        summary["MatchRateByType"] = triggered_df.groupby(["Type"]).agg({"Match":"mean"}).to_dict()["Match"] \
            if len(triggered_df.columns) else None
        summary["SuggestionsShownByWordCount"] = triggered_df.groupby(["WordCount"]).size().to_dict() \
            if len(triggered_df.columns) else None
        summary["SuggestionsMatchedByWordCount"] = triggered_df[triggered_df["Match"]].groupby(["WordCount"]).size().to_dict() \
            if len(triggered_df.columns) else None
        summary["MatchRateByWordCount"] = triggered_df.groupby(["WordCount"]).agg({"Match":"mean"}).to_dict()["Match"] \
            if len(triggered_df.columns) else None
        return summary

    def score(self, min_scores: list|float, expected_match_rate: float = None) -> list:
        """Score the text prediction sequence.

        This function extracts predictions, calculates which one will be triggered at specific min_scores
        and can estimate score for expected match rate.

        Args:
            min_scores (list|float): min_score to show the predictions
            expected_match_rate (float, optional): If given, estimate score for the given expected match rate.
            (provided as float from 0 to 1)

        Returns:
            list: Summary list with all the metrics
        """
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
                match_rates = [summ["MatchRate"] for summ in summary_list]
                scores = [summ["Score"] for summ in summary_list]
                if len(match_rates) < 2:
                    logging.warning("Not enough points to calculate score for the expected match rate of %.3f", expected_match_rate)
                    continue
                f = interp1d(match_rates, scores, bounds_error=False, kind="linear", fill_value="extrapolate")
                # f = np.poly1d(np.polyfit(match_rates, scores, 2))
                min_score = float(f(expected_match_rate))
                logging.debug("Expected_score = %s at %s", min_score, expected_match_rate)

            triggered_df = self.calc_triggered_predictions(min_score, predictions_df)
            self._triggered_df = triggered_df
            summary_dict = self.score_triggered_predictions(triggered_df, min_score)
            summary_list.append(summary_dict)

        self._score_summary_list = summary_list
        return summary_list
