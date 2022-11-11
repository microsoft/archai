from functools import partial
from typing import Tuple, Optional

from overrides import overrides
import tensorwatch as tw

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.datasets.dataset_provider import DatasetProvider
from archai.discrete_search.api.objective import Objective


class TensorwatchStat(Objective):
    def __init__(self, stat_name: str, input_shape: Tuple):
        """Calculates an evaluation statistic for a PyTorch model using Tensorwatch.

        Args:
            stat_name (str): Name of the statistic. Available options ['Madd', 'Flops', 'parameters', 
                'mwrite', 'mread', 'duration', 'inference_memory'].
            
            input_shape (Tuple): _description_
        """
        self.stat_name = stat_name
        self.input_shape = input_shape
        self.higher_is_better = False

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider, 
                budget: Optional[float] = None) -> float:
        model_stats = tw.ModelStats(model.arch, self.input_shape, clone_model=True)
        return model_stats.__dict__[self.stat_name]


TorchMadds = partial(TensorwatchStat, stat_name='MAdd')
TorchFlops = partial(TensorwatchStat, stat_name='Flops')
TorchNumParameters = partial(TensorwatchStat, stat_name='parameters')
