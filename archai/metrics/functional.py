from typing import Callable
from overrides import overrides
from torch.utils.data import DataLoader

from archai.metrics.base import BaseMetric
from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider


class FunctionalMetric(BaseMetric):
    def __init__(self, evaluation_fn: Callable, higher_is_better: bool = False, 
                 subset: str = 'test', batch_size: int = 32, **dataloader_kwargs):
        """Uses the result of a custom function as a metric.

        Args:
            evaluation_fn (Callable): Evaluation function that receives the parameters
                (model: ArchWithmetadata, dataloader: torch.utils.data.Dataloader) and outputs a float.
            higher_is_better (bool): Optimization direction. True for maximization, False for minimization.
            subset (str, optional): Dataset subset ('train', 'test', 'validation'). Defaults to 'test'.
            batch_size (int, optional): Dataloader batch size. Defaults to 32
        """        
        self.evaluation_fn = evaluation_fn
        self.higher_is_better = higher_is_better
        
        assert subset in ['train', 'test', 'validation']
        self.subset = subset
        self.batch_size = batch_size
        self.dl_kwargs = dataloader_kwargs

    @overrides
    def compute(self, model: ArchWithMetaData, dataset_provider: DatasetProvider) -> float:
        if self.subset == 'train':
            dataset, _ = dataset_provider.get_datasets(True, False, None, None)
        elif self.subset == 'test':
            _, dataset = dataset_provider.get_datasets(False, True, None, None)
        else:
            raise NotImplementedError

        dl = DataLoader(dataset, batch_size=self.batch_size, **self.dl_kwargs)
        return self.evaluation_fn(model, dl)
