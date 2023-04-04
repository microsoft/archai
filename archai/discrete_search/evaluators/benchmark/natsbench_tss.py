# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import Any, Dict, Optional

import nats_bench
from overrides import overrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.search_spaces.benchmark.natsbench_tss import (
    NatsbenchTssSearchSpace
)


class NatsbenchMetric(ModelEvaluator):
    """Evaluate a model using a metric from the NATS-Bench API."""

    def __init__(
        self,
        search_space: NatsbenchTssSearchSpace,
        metric_name: str,
        epochs: Optional[int] = None,
        raise_not_found: Optional[bool] = True,
        more_info_kwargs: Optional[Dict[str, Any]] = None,
        cost_info_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            search_space: Search space to use.
            metric_name: Metric to use. See `nats_bench.api.NatsBenchAPI.get_more_info` for available metrics.
            epochs: Number of epochs to use. If None, uses the default number of epochs.
            raise_not_found: If True, raises an error if the architecture does not belong to the search space.
            more_info_kwargs: Additional arguments to pass to `nats_bench.api.NatsBenchAPI.get_more_info`.
            cost_info_kwargs: Additional arguments to pass to `nats_bench.api.NatsBenchAPI.get_cost_info`.

        """

        assert isinstance(
            search_space, NatsbenchTssSearchSpace
        ), "This objective function only works with architectures from NatsbenchTssSearchSpace"

        self.search_space = search_space
        self.metric_name = metric_name
        self.epochs = epochs

        self.archid_pattern = re.compile("natsbench-tss-([0-9]+)")
        self.api = nats_bench.create(str(self.search_space.natsbench_location), "tss", fast_mode=True, verbose=False)

        self.raise_not_found = raise_not_found
        self.more_info_kwargs = more_info_kwargs or dict()
        self.cost_info_kwargs = cost_info_kwargs or dict()
        self.total_time_spent = 0

    @overrides
    def evaluate(self, model: ArchaiModel, budget: Optional[float] = None) -> Optional[float]:
        natsbench_id = self.archid_pattern.match(model.archid)
        budget = int(budget) if budget else budget

        if not natsbench_id:
            if self.raise_not_found:
                raise ValueError(
                    f"Architecture {model.archid} does not belong to the NatsBench search space. "
                    "Please refer to `archai.search_spaces.discrete.NatsbenchSearchSpace` to "
                    "use the Natsbench search space."
                )

            return None

        info = self.api.get_more_info(
            int(natsbench_id.group(1)),
            dataset=self.search_space.base_dataset,
            iepoch=budget or self.epochs,
            **self.more_info_kwargs,
        )

        cost_info = self.api.get_cost_info(
            int(natsbench_id.group(1)), dataset=self.search_space.base_dataset, **self.cost_info_kwargs
        )

        if self.metric_name in info:
            result = info[self.metric_name]
            self.total_time_spent += info["train-all-time"] + info["test-all-time"]
        elif self.metric_name in cost_info:
            result = info[self.metric_name]
        else:
            raise KeyError(f"`metric_name` {self.metric_name} not found. Available metrics = {str(list(info.keys()))}")

        return result
