# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Text Predict-based predictions and sequences."""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ftfy
import numpy as np
import pandas as pd
from tqdm import tqdm

from archai.nlp.eval.eval_utils import cached_property
from archai.nlp.eval.text_predict.text_predict_model import TextPredictModel
from archai.nlp.eval.text_predict.text_predict_tokenizer import TextPredictTokenizer


class TextPredictPrediction:
    """Single prediction from Text Predict."""

    # Constants for P(Accept) calculation
    a = 0.04218
    b = -0.1933

    def __init__(
        self,
        text: str,
        probability: float,
        score: Optional[float] = None,
        input_ids: Optional[Tuple[int, ...]] = None,
        token_ids: Optional[Tuple[int, ...]] = None,
        end_with_complete_word: Optional[bool] = None,
        has_matched: Optional[bool] = None,
    ) -> None:
        """Override initialization method.

        Args:
            text: The predicted text.
            probability: The probability of the prediction.
            score: The score of the prediction.
            input_ids: The input identifiers.
            token_ids: The token identifiers.
            end_with_complete_word: Whether the prediction ends with a complete word.
            has_matched: Whether the prediction has matched the reference.

        """

        self.text = text

        self.probability = probability
        self.score = score

        self.input_ids = input_ids
        self.token_ids = token_ids

        self.end_with_complete_word = end_with_complete_word
        self.has_matched = has_matched

    def __len__(self) -> int:
        """Return the length of the prediction.

        Returns:
            Prediction length.

        """

        return len(str(self))

    def __str__(self) -> str:
        """Return the string representation of the prediction.

        Returns:
            String representation.

        """

        return self.text.rstrip()

    def __repr__(self) -> str:
        """Return the print representation of the prediction.

        Returns:
            Print representation.

        """

        return f"({self.text}, {self.probability:.5f}, {self.score():.3f})"

    def p_match(self) -> float:
        """Return the probability of the prediction being matched.

        Returns:
            Match probability.

        """

        return self.probability

    def p_accept(self) -> float:
        """Return the probability of the prediction being accepted.

        Returns:
            Accept probability.

        """

        result = self.probability * self.p_accept_given_match()

        if result < 0:
            return 0.0

        return result

    def p_accept_given_match(self) -> float:
        """Return the probability of the prediction being accepted given a match.

        Returns:
            Accept probability given match.

        """

        result = self.a * len(self) + self.b

        if result < 0:
            return 0.0

        return result

    def p_char_accept(self) -> float:
        """Return the probability of the characters from the prediction being accepted.

        Returns:
            Character accept probability.

        """

        return len(self) * self.p_accept()

    def all_ids(self) -> Tuple[int, ...]:
        """Return the combined `input_ids` and `token_ids` of the prediction.

        Returns:
            Combined `input_ids` and `token_ids`.

        """

        if self.input_ids is None or self.token_ids is None:
            raise ValueError(f"Unable to determine `all_ids` for `{self}`.")

        return self.input_ids + self.token_ids

    def length_type(self) -> str:
        """Return the type of the prediction based on its length.

        Returns:
            Type based on prediction length.

        """

        pred_length = len(self)

        if pred_length < 6:
            return "0:XS"

        if pred_length < 11:
            return "1:S"

        if pred_length < 16:
            return "2:M"

        return "3:L"

    def word_count(self) -> int:
        """Return the number of words in the prediction.

        Returns:
            Amount of words in prediction.

        """

        if len(self) == 0:
            return 0

        return len(re.findall(r"\s+", self.text.strip())) + 1

    def is_empty(self) -> bool:
        """Return whether prediction is empty or not.

        Returns:
            Whether prediction is empty or not.

        """

        return len(self.text) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Return meta-information about the prediction.

        Returns:
            Meta-information about prediction.

        """

        return {
            "Text": str(self),
            "Probability": self.probability,
            "Length": len(self),
            "EndWithCompleteWord": self.end_with_complete_word,
            "Match": self.has_matched,
            "PAccept": self.p_accept(),
            "Score": self.score(),
            "CharAccepted": self.p_char_accept(),
            "WordCount": self.word_count(),
            "Tokens": self.token_ids,
        }

    @classmethod
    def empty(cls: TextPredictPrediction) -> TextPredictPrediction:
        """Create an empty prediction.

        Returns:
            Empty prediction.

        """

        return TextPredictPrediction("", 0.0, end_with_complete_word=False)

    @classmethod
    def next_prediction(
        cls: TextPredictPrediction,
        prediction: TextPredictPrediction,
        text: str,
        probability: float,
        token_id: int,
    ) -> TextPredictPrediction:
        """Create the next prediction given a previous prediction.

        Args:
            prediction: The previous prediction.
            text: The next prediction text.
            probability: The next prediction probability.
            token_id: The next prediction token identifier.

        Returns:
            Next prediction.

        """

        return TextPredictPrediction(
            prediction.text + text,
            prediction.probability * probability,
            input_ids=prediction.input_ids,
            token_ids=prediction.token_ids + (token_id,),
        )


@dataclass
class TextPredictionPosition:
    """Represents the position from a Text Predict-based prediction.

    Args:
        line_id: The line identifier.
        char_id: The character identifier.
        body: The body (context) of information.
        body_continued: The continued body (reference) of information.
        prediction: The prediction for this position.
        time: The time spent to calculate the prediction.

    """

    line_id: str = field(metadata={"help": "Line identifier."})

    char_id: str = field(metadata={"help": "Character identifier."})

    body: str = field(metadata={"help": "Body (context) of information."})

    body_continued: str = field(metadata={"help": "Continued body (reference) of information."})

    prediction: TextPredictPrediction = field(default=None, metadata={"help": "Prediction."})

    time: int = field(default=None, metadata={"help": "Time spent to calculate prediction."})

    @property
    def unique_id(self) -> str:
        """Return the unique identifier of position.

        Returns:
            Unique identifier.

        """

        return f"{self.line_id}-{self.char_id}"

    @classmethod
    def from_ljson(cls: TextPredictionPosition, line: str) -> TextPredictionPosition:
        """Load a position from a .ljson file.

        Args:
            line: Encoded line with .ljson.

        Returns:
            Text Predict-based position.

        """

        d = json.loads(line)

        unique_ids = d["UniqueId"].split("-")
        if len(unique_ids) > 2:
            raise ValueError("Unable to split UniqueIds `{unique_ids}` into LineId and CharId.")

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
            prediction = TextPredictPrediction(text, probability, score=prediction_score)
        else:
            prediction = None

        return TextPredictionPosition(
            line_id=line_id,
            char_id=char_id,
            body=d["Body"],
            body_continued=d["BodyContinued"],
            prediction=prediction,
            time=d.get("Time", None),
        )

    def to_ljson(self) -> str:
        """Convert position to a .ljson encoded string.

        Returns:
            Encoded position with .ljson.

        """

        output = OrderedDict(
            {
                "UniqueId": self.unique_id,
                "Body": self.body,
                "BodyContinued": self.body_continued,
            }
        )

        if self.time is not None:
            output["Time"] = self.time

        if self.prediction is None:
            output["Suggestions"] = []
        else:
            output["Suggestions"] = [
                {
                    "Suggestion": str(self.prediction),
                    "Probability": self.prediction.probability,
                    "Score": self.prediction.score(),
                }
            ]

        return json.dumps(output)


class TextPredictionSequence(OrderedDict):
    """Represents a sequence of positions inside the Text Prediction pipeline."""

    def __init__(
        self,
        max_body_length: Optional[int] = 1000000,
        save_step: Optional[int] = 100000,
        current_paragraph_only: Optional[bool] = False,
        min_score: Optional[float] = 1.0,
        min_pred_length: Optional[int] = 6,
    ):
        """Overrides initialization method.

        Args:
            max_body_length: Maximum length of the input text.
            save_step: Amount of steps to save results.
            current_paragraph_only: Only predicts information from current paragraph.
            min_score: Minimum score.
            min_pred_length: Minimum length of the prediction.

        """

        super().__init__()

        self.max_body_length = max_body_length
        self.save_step = save_step
        self.current_paragraph_only = current_paragraph_only
        self.min_score = min_score
        self.min_pred_length = min_pred_length

        self.perplexity = None
        self.score_summary = None
        self.triggered_preds = None

    @classmethod
    def from_file(
        cls: TextPredictionSequence,
        file_path: str,
        **kwargs,
    ) -> TextPredictionSequence:
        """Load a sequence (list of positions) from a file.

        Args:
            file_path: The path to a file with the sequence data.

        Returns:
            Instance of sequence.

        """

        file_type = os.path.splitext(file_path)[1]
        if file_type == ".ljson":
            return TextPredictionSequence.from_ljson_file(file_path, **kwargs)
        if file_type == ".txt":
            return TextPredictionSequence.from_text_file(file_path, **kwargs)

        raise NotImplementedError

    @classmethod
    def from_ljson_file(
        cls: TextPredictionSequence,
        file_path: str,
        **kwargs,
    ) -> TextPredictionSequence:
        """Load a sequence (list of positions) from a .ljson file.

        Args:
            file_path: The path to a file with the sequence data.

        Returns:
            Instance of sequence.

        """

        lines = []
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        sequence = TextPredictionSequence(**kwargs)
        for line in lines:
            position = TextPredictionPosition.from_ljson(line)
            sequence[position.unique_id] = position

        return sequence

    @classmethod
    def from_text_file(
        cls: TextPredictionSequence,
        file_path: str,
        new_document_regex: Optional[str] = "\\n\\n+",
        **kwargs,
    ) -> TextPredictionSequence:
        """Load a sequence (list of positions) from a .txt file.

        Args:
            file_path: The path to a file with the sequence data.
            new_document_regex: The regex to identify a new document.

        Returns:
            Instance of sequence.

        """

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        lines = re.split(new_document_regex, text, flags=re.DOTALL | re.MULTILINE)
        sequence = TextPredictionSequence(**kwargs)

        for line_id, line in enumerate(lines):
            line = line.strip()
            line = ftfy.fix_text(line)

            for char_id in range(len(line)):
                position = TextPredictionPosition(
                    line_id=line_id,
                    char_id=char_id,
                    body=line[:char_id],
                    body_continued=line[char_id:],
                    prediction=None,
                    time=None,
                )
                sequence[position.unique_id] = position

        return sequence

    def _filter_keys_char_id(self, char_id: Optional[int] = None) -> List[str]:
        """Filter keys based on the character identifier.

        Args:
            char_id: The character identifier.

        Returns:
            A list of filtered keys.

        """

        return [k for k, v in self.items() if v.char_id == char_id]

    @cached_property
    def word_count(self) -> int:
        """Calculate the word count of sequence.

        Returns:
            The word count.

        """

        word_count = 0
        for unique_id in self._filter_keys_char_id(1):
            text = self[unique_id].body + self[unique_id].body_continued
            word_count += len(re.findall(r"[^\s]+", text, re.DOTALL | re.MULTILINE))

        return word_count

    def save(self, output_dir: str) -> None:
        """Save predictions, scoring summary and triggered predictions to output files.

        Args:
            output_dir: The output directory.

        """

        PREDICTION_FILE = "preds.ljson"
        SUMMARY_FILE = "summary.json"
        TRIGGERED_FILE = "triggered_preds.csv"

        os.makedirs(output_dir, exist_ok=True)

        prediction_file_path = os.path.join(output_dir, PREDICTION_FILE)
        preds = [pos.to_ljson() + "\n" for pos in self.values()]
        with open(prediction_file_path, "w") as f:
            f.writelines(preds)

        summary_file_path = os.path.join(output_dir, SUMMARY_FILE)
        with open(summary_file_path, "w") as f:
            summary = json.dumps(self.score_summary, indent=1)
            f.write(summary)

        if self.triggered_preds is not None:
            triggered_prediction_file_path = os.path.join(output_dir, TRIGGERED_FILE)
            self.triggered_preds.to_csv(triggered_prediction_file_path, index=False)

    def get_predictions(self) -> pd.DataFrame:
        """Convert instance into a data frame of predictions.

        Returns:
            A data frame of predictions.

        """

        predictions = []
        for pos in self.values():
            prediction = pos.prediction

            if prediction is not None:
                body_continued = pos.body_continued[: len(prediction)]
                prediction.has_matched = str(prediction) == body_continued

                min_length = min(len(str(prediction)), len(body_continued))
                last_matched_char = next(
                    (i for i in range(min_length) if str(prediction)[i] != body_continued[i]),
                    min_length,
                )

                length_type = prediction.length_type()
                prediction_odict = prediction.to_odict()
                p_accept_given_match = prediction.p_accept_given_match()
            else:
                body_continued, min_length, last_matched_char = True, 0, 0

                prediction_odict = OrderedDict(
                    [
                        ("Text", ""),
                        ("Probability", 0.0),
                        ("Length", 0),
                        ("EndWithCompleteWord", False),
                        ("Match", None),
                        ("PAccept", 0.0),
                        ("Score", 0.0),
                        ("CharAccepted", 0.0),
                        ("WordCount", 0),
                        ("Tokens", None),
                    ]
                )
                length_type = ""
                p_accept_given_match = 0.0

            prediction_odict["Line"] = pos.line_id
            prediction_odict["Char"] = pos.char_id
            prediction_odict["BodyContinued"] = body_continued
            prediction_odict["Type"] = length_type
            prediction_odict["LastMatchChar"] = last_matched_char
            prediction_odict["NextTrigger"] = pos.char_id + last_matched_char + 1
            prediction_odict["PAcceptGivenMatch"] = p_accept_given_match

            predictions.append(prediction_odict)

        predictions_df = pd.DataFrame(predictions)
        predictions_df_columns = ["Line", "Char", "Text", "BodyContinued"]
        predictions_df_columns = (
            predictions_df_columns
            + predictions_df.columns.drop(predictions_df_columns + ["EndWithCompleteWord", "Tokens"]).tolist()
        )
        predictions_df = predictions_df[predictions_df_columns]

        return predictions_df

    def calculate_triggered_predictions(
        self, min_score: float, predictions_df: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Calculate the triggered predictions.

        Args:
            min_score: The minimum score.
            predictions_df: The predictions data frame.

        Returns:
            A data frame of triggered predictions.

        """

        triggered_predictions = []

        if predictions_df is None:
            predictions_df = self.get_predictions()

        # Keeps count where prediction has triggered
        line_id = -1
        char_id = -1

        # Triggered is an array that denotes if suggestion is regarded as 'triggered'
        # -1: score was too low
        #  0: score OK, but something is being shown and current suggestion could not be shown
        #  1: suggestion shown
        triggered = np.full((len(predictions_df.index),), -1)

        score_values = predictions_df["Score"].values
        for i in tqdm(range(predictions_df.shape[0]), desc=f"Scoring with {min_score:.2f}"):
            if score_values[i] < min_score:
                continue

            if predictions_df["Line"][i] < line_id:
                msg = f"Incorrect order of lines in the file (current line = {line_id}, "
                msg += f'processed line = {predictions_df["Line"][i]}; current char = {char_id}, '
                msg += f'processed char = {predictions_df["Char"][i]}'

                raise ValueError(msg)

            triggered[i] = 0
            if predictions_df["Line"][i] > line_id or predictions_df["Char"][i] > char_id:
                d = predictions_df.iloc[i].to_dict()

                line_id = d["Line"]
                char_id = d["Char"] + d["LastMatchChar"]

                triggered_predictions.append(d)
                triggered[i] = 1

        triggered_df = pd.DataFrame(triggered_predictions)
        triggered_column_name = f"Trigger: {min_score}"
        predictions_df[triggered_column_name] = triggered

        return triggered_df

    def calculate_perplexity(self, model: TextPredictModel, tokenizer: TextPredictTokenizer) -> float:
        """Calculate the perplexity of sequence.

        Args:
            model: Text Predict-based model used to calculate the perplexity.
            tokenizer: Text Predict-based tokenizer used to calculate the perplexity.

        Returns:
            The perplexity of the sequence.

        """

        loss = 0.0
        token_ids_length = 0

        for unique_id in self._filter_keys_char_id(1):
            text = self[unique_id].body + self[unique_id].body_continued
            text = tokenizer.clean_text(text)

            token_ids = tokenizer.encode(text)
            token_ids_length += len(token_ids)

            loss += len(token_ids) * model.get_loss(tuple(token_ids))

        perplexity = np.exp(loss / token_ids_length)

        return perplexity

    def score_triggered_predictions(
        self,
        triggered_df: pd.DataFrame,
        model: TextPredictModel,
        tokenizer: TextPredictTokenizer,
        min_score: Optional[float] = None,
    ) -> OrderedDict:
        """Score the triggered predictions.

        Args:
            triggered_df: The triggered predictions data frame.
            min_score: The minimum score.
            model: Text Predict-based model used to calculate the perplexity.
            tokenizer: Text Predict-based tokenizer used to calculate the perplexity.

        Returns:
            A dictionary of the triggered predictions and their scores.

        """

        # Allows perplexity to be cached and avoids re-computation
        if self.perplexity is None:
            self.perplexity = self.calculate_perplexity(model, tokenizer)

        summary = OrderedDict()
        summary["Score"] = min_score
        summary["TotalEvalPoints"] = len(self)
        summary["TotalWordCount"] = self.word_count
        summary["Perplexity"] = self.perplexity
        summary["SuggestionsShown"] = len(triggered_df.index)
        summary["SuggestionsMatched"] = int(np.sum(triggered_df["Match"])) if len(triggered_df.columns) else 0
        summary["SuggestionsAccepted"] = (
            int(np.sum(triggered_df["Match"] * triggered_df["PAcceptGivenMatch"])) if len(triggered_df.columns) else 0
        )
        summary["SuggestionRatePerWord"] = summary["SuggestionsShown"] / summary["TotalWordCount"]
        summary["SuggestionRatePerChar"] = summary["SuggestionsShown"] / summary["TotalEvalPoints"]
        summary["MatchRate"] = np.mean(triggered_df["Match"]) if len(triggered_df.columns) else 0
        summary["AcceptRate"] = (
            np.mean(triggered_df["Match"] * triggered_df["PAcceptGivenMatch"]) if len(triggered_df.columns) else 0
        )
        summary["CharMatched"] = (
            int(np.sum(triggered_df["Match"] * triggered_df["Length"])) if len(triggered_df.columns) else 0
        )
        summary["CharAccepted"] = (
            int(np.sum(triggered_df["Match"] * triggered_df["PAcceptGivenMatch"] * triggered_df["Length"]))
            if len(triggered_df.columns)
            else 0
        )
        summary["CharMatchRate"] = summary["CharMatched"] / summary["TotalEvalPoints"]
        summary["CharAcceptRate"] = summary["CharAccepted"] / summary["TotalEvalPoints"]
        summary["SuggestionsShownByType"] = (
            triggered_df.groupby(["Type"]).size().to_dict() if len(triggered_df.columns) else None
        )
        summary["SuggestionsMatchedByType"] = (
            triggered_df[triggered_df["Match"]].groupby(["Type"]).size().to_dict() if len(triggered_df.columns) else 0
        )
        summary["MatchRateByType"] = (
            triggered_df.groupby(["Type"]).agg({"Match": "mean"}).to_dict()["Match"]
            if len(triggered_df.columns)
            else None
        )
        summary["SuggestionsShownByWordCount"] = (
            triggered_df.groupby(["WordCount"]).size().to_dict() if len(triggered_df.columns) else None
        )
        summary["SuggestionsMatchedByWordCount"] = (
            triggered_df[triggered_df["Match"]].groupby(["WordCount"]).size().to_dict()
            if len(triggered_df.columns)
            else None
        )
        summary["MatchRateByWordCount"] = (
            triggered_df.groupby(["WordCount"]).agg({"Match": "mean"}).to_dict()["Match"]
            if len(triggered_df.columns)
            else None
        )

        return summary
