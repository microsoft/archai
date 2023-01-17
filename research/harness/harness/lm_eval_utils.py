# Copyright: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py

import fnmatch
from typing import List


def pattern_match(patterns: List[str], source_list: List[str]) -> List[str]:
    task_names = set()

    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)

    return list(task_names)


class MultiChoice:
    def __init__(self, choices: List[str]):
        self.choices = choices

    def __contains__(self, values: List[str]) -> bool:
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self) -> str:
        for choice in self.choices:
            yield choice
