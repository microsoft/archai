import random
from typing import List
from overrides.overrides import overrides

from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.algos.evolution_pareto_image_seg.segmentation_search_space import SegmentationSearchSpace

class DiscreteSearchSpaceSegmentation(DiscreteSearchSpace):
    def __init__(self, datasetname:str):
        super().__init__()
        self.datasetname = datasetname
        

    @overrides
    def random_sample(self)->ArchWithMetaData:
        ''' Uniform random sample an architecture '''
        model = SegmentationSearchSpace.random_sample(nb_layers=12, 
                                                    max_downsample_factor=16, 
                                                    max_scale_delta=1)

        meta_data = {
            'datasetname': self.datasetname 
        }
        arch_meta = ArchWithMetaData(model, meta_data)
        return arch_meta
        
