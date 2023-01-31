# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from evaluate import load
from evaluate.evaluation_suite import EvaluationSuite, SubTask


class GLUESuite(EvaluationSuite):
    def __init__(self) -> None:
        super().__init__("glue-tasks")

        self.preprocessor = lambda x: {"text": x["text"].lower()}
        self.suite = [
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="cola",
                split="validation",
                args_for_task={
                    "metric": load("glue", "cola"),
                    "input_column": "sentence",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="mnli",
                split="validation_matched",
                args_for_task={
                    "metric": load("glue", "mnli_matched"),
                    "input_column": "premise",
                    "second_input_column": "hypothesis",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="mnli",
                split="validation_mismatched",
                args_for_task={
                    "metric": load("glue", "mnli_mismatched"),
                    "input_column": "premise",
                    "second_input_column": "hypothesis",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="mrpc",
                split="test",
                args_for_task={
                    "metric": load("glue", "mrpc"),
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="qnli",
                split="validation",
                args_for_task={
                    "metric": load("glue", "qnli"),
                    "input_column": "question",
                    "second_input_column": "sentence",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="qqp",
                split="validation",
                args_for_task={
                    "metric": load("glue", "qqp"),
                    "input_column": "question1",
                    "second_input_column": "question2",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="rte",
                split="validation",
                args_for_task={
                    "metric": load("glue", "rte"),
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="sst2",
                split="validation",
                args_for_task={
                    "metric": load("glue", "sst2"),
                    "input_column": "sentence",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1},
                },
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="wnli",
                split="validation",
                args_for_task={
                    "metric": load("glue", "wnli"),
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0, "LABEL_1": 1},
                },
            ),
        ]
