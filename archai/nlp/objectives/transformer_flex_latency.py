from overrides import overrides
from typing import Optional

from archai.discrete_search import ArchaiModel, Objective, DatasetProvider
from archai.nlp.compression.onnx_constraints import *


class TransformerFlexTorchLatency(Objective):
    higher_is_better: bool = False

    def __init__(self) -> None:
        super().__init__()
    
    @overrides
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider,
                 budget: Optional[float] = None) -> float:
        return super().evaluate(arch, dataset, budget)


class TransformerFlexOnnxLatency(Objective):
    higher_is_better: bool = False

    def __init__(self) -> None:
        super().__init__()
    
    @overrides
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider,
                 budget: Optional[float] = None) -> float:
        return super().evaluate(arch, dataset, budget)

