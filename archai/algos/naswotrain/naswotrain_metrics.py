from archai.common.metrics import Metrics

from overrides import overrides, EnforceOverrides
from typing import List, Mapping, Optional, Tuple

from archai.common.apex_utils import ApexUtils


class NaswoTrainMetrics(Metrics, EnforceOverrides):
    def __init__(self, title: str, apex: Optional[ApexUtils], logger_freq: int) -> None:
        super().__init__(title, apex, logger_freq=logger_freq)
        self._naswotraining_score = -1000.0

    @property
    def naswotraining_score(self)-> float:
        return self._naswotraining_score

    @naswotraining_score.setter
    def naswotraining_score(self, score:float)-> None:
        self._naswotraining_score = score
