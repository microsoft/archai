from functools import partial
from typing import Tuple, Optional

from overrides import overrides
import tensorwatch as tw

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric


class TensorwatchMetric(BaseMetric):
    def __init__(self, metric_name: str, input_shape: Tuple):
        """Calculates a metric for a PyTorch model using Tensorwatch.

        Args:
            metric_name (str): Name of the metric. Available options ['Madd', 'Flops', 'parameters', 
                'mwrite', 'mread', 'duration', 'inference_memory'].
            input_shape (Tuple): _description_
        """
        self.metric_name = metric_name
        self.input_shape = input_shape
        self.higher_is_better = False

    @overrides
    def compute(self, model: ArchWithMetaData, dataset_provider: DatasetProvider, 
                budget: Optional[float] = None) -> float:
        model_stats = tw.ModelStats(model.arch, self.input_shape, clone_model=True)
        return model_stats.__dict__[self.metric_name]


MaddsMetric = partial(TensorwatchMetric, metric_name='MAdd')
FlopsMetric = partial(TensorwatchMetric, metric_name='Flops')
NumParametersMetric = partial(TensorwatchMetric, metric_name='parameters')
