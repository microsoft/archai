# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Report-like entrypoint that is capable of parsing logs and producing
    human-readable information, such as tables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from archai.nlp.reports.dllogger_parser import parse_json_dlogger_file


class ArchaiReport:
    """Base report class that provides utilities to load dllogger files and
        create human-readable outputs.

    """

    def __init__(self,
                 train_logs: Dict[str, Dict[str, Any]],
                 test_logs: Dict[str, Any]) -> None:
        """Initializes the class by cleaning up the input logs.

        Args:
            train_logs: Training-related logs retrieved from a dllogger file.
            test_logs: Test-related logs retrieved from a dllogger file.
            
        """

        # Cleans up the training logs by converting to non-nested dictionaries
        self.train_logs = self._clean_nested_dict(train_logs)
        self.test_logs = test_logs

    @classmethod
    def from_json_file(cls: ArchaiReport, json_file: str) -> ArchaiReport:
        train_logs, test_logs = parse_json_dlogger_file(json_file)

        return cls(train_logs, test_logs)

    def _clean_nested_dict(self, input_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_list = [value for value in input_dict.values()]

        return output_list

    def create_markdown(self) -> Tuple[str, str]:
        # Creates dataframes from the available dictionaries
        train_logs_df = pd.DataFrame(self.train_logs)
        test_logs_df = pd.DataFrame(self.test_logs, index=[0])

        return train_logs_df.to_markdown(index=False), test_logs_df.to_markdown(index=False)
