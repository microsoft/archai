import random
from typing import List
from overrides.overrides import overrides

from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.algos.natsbench.natsbench_utils import create_natsbench_tss_api, model_from_natsbench_tss

class DiscreteSearchSpaceNatsbenchTSS(DiscreteSearchSpace):
    def __init__(self, datasetname:str, natsbench_location:str):
        super().__init__()
        self.datasetname = datasetname
        self.natsbench_location = natsbench_location
        
        # Natsbench TSS ops bag
        self.OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']

        # create the natsbench api
        self.api = create_natsbench_tss_api(self.natsbench_location)

    @overrides
    def random_sample(self)->ArchWithMetaData:
        ''' Uniform random sample an architecture '''
        curr_archid = random.sample(range(len(self.api)), k=1)[0]

        # if not in cache actually evaluate it
        model = model_from_natsbench_tss(curr_archid, self.datasetname, self.api)

        meta_data = {
            'datasetname': self.datasetname,
            'archid': curr_archid
        }
        arch_meta = ArchWithMetaData(model, meta_data)
        return arch_meta


    @overrides
    def get_neighbors(self, arch: ArchWithMetaData) -> List[ArchWithMetaData]:
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # first get the string representation of the current architecture
        archid = arch.metadata['archid']
        string_rep = self.api.get_net_config(archid, self.datasetname)['arch_str']
        nbhd_strs = []
        ops = self._get_op_list(string_rep)
        for i in range(len(ops)):
            available = [op for op in self.OPS if op != ops[i]]
            for op in available:
                new_ops = ops.copy()
                new_ops[i] = op
                new_arch_str = self._get_string_from_ops(new_ops)
                nbhd_strs.append(new_arch_str)

        # convert the arch strings to architecture ids
        nbhd_archs = []
        for arch_str in nbhd_strs:
            this_archid = self.api.archstr2index[arch_str]
            model = model_from_natsbench_tss(this_archid, self.datasetname, self.api)
            meta_data = {
                'datasetname': self.datasetname,
                'archid': this_archid
            }
            arch_meta = ArchWithMetaData(model, meta_data)
            nbhd_archs.append(arch_meta)
        return nbhd_archs


    def _get_op_list(self, string:str)->List[str]:
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # given a string, get the list of operations
        tokens = string.split('|')
        ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
        return ops