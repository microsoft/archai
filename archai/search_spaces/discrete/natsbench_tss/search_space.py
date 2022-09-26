import random
import re
import yaml
import warnings
from pathlib import Path
from typing import List
from overrides import overrides

import nats_bench

from archai.nas.arch_meta import ArchWithMetaData
from archai.search_spaces.discrete.base import EvolutionarySearchSpaceBase
from archai.algos.natsbench.natsbench_utils import model_from_natsbench_tss


class NatsbenchTssSearchSpace(EvolutionarySearchSpaceBase):
    # Natsbench TSS valid operations
    OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']

    def __init__(self, natsbench_location: str, base_dataset: str) -> None:
        self.natsbench_location = Path(natsbench_location)
        self.base_dataset = base_dataset
        assert base_dataset in ['cifar10', 'cifar100', 'ImageNet16-120'], \
            "`base_dataset` must be one of ['cifar10', 'cifar100', 'ImageNet16-120']" 

        if not self.natsbench_location.exists():
            raise FileNotFoundError(
                'The provided path to `natsbench_location` ('
                f'{self.natsbench_location.absolute()}) does not exist'
            )
    
        self.api = nats_bench.create(
            natsbench_location, 'tss', fast_mode=True, verbose=False
        )

        self.archid_pattern = re.compile(f'natsbench-tss-([0-9]+)')

    def _get_op_list(self, string:str) -> List[str]:
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # Given a string, get the list of operations
        tokens = string.split('|')
        ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
        
        return ops

    def _get_string_from_ops(self, ops):
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # Given a list of operations, get the string
        
        strings = ['|']
        nodes = [0, 0, 1, 0, 1, 2]
        
        for i, op in enumerate(ops):
            strings.append(op+'~{}|'.format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i+1] == 0:
                strings.append('+|')
        return ''.join(strings)

    @overrides
    def save_arch(self, model: ArchWithMetaData, path: str) -> None:
        yaml.safe_dump(model.metadata, open(path, 'w', encoding='utf-8'))

    @overrides
    def load_arch(self, path: str) -> ArchWithMetaData:
        metadata = yaml.safe_load(open(path, encoding='utf-8'))
        natsbenchid = self.archid_pattern.match(metadata['archid'])

        if not natsbenchid:
            raise ValueError(
                f'Architecture {metadata["archid"]} does not belong to the `NatsbenchTssSearchSpace`. '
            )

        if metadata['dataset'] != self.base_dataset:
            warnings.warn(
                f'Architecture loaded from {path} was saved using a different dataset ({metadata["dataset"]})'
                f' than `NatsbenchTssSearchSpace` base dataset ({self.base_dataset})'
            )

        return self.get([int(natsbenchid.group(1))])

    @overrides
    def get(self, idx_vector: List[int]) -> ArchWithMetaData:
        idx = idx_vector[0] % len(self.api)
        
        return ArchWithMetaData(
            model=model_from_natsbench_tss(idx, self.base_dataset, self.api),
            extradata={'archid': f'natsbench-tss-{idx}', 'dataset': self.base_dataset}
        )

    @overrides
    def mutate(self, arch: ArchWithMetaData) -> ArchWithMetaData:
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # First get the string representation of the current architecture
        archid = arch.metadata['archid']
        natsbenchid = self.archid_pattern.match(archid)

        if not natsbenchid:
            raise ValueError(
                f'Architecture {archid} does not belong to the `NatsbenchTssSearchSpace`. '
            )

        natsbenchid = int(natsbenchid.group(1))
        string_rep = self.api.get_net_config(natsbenchid, self.base_dataset)['arch_str']

        nbhd_strs = []
        ops = self._get_op_list(string_rep)
        
        for i in range(len(ops)):
            available = [op for op in self.OPS if op != ops[i]]

            for op in available:
                new_ops = ops.copy()
                new_ops[i] = op
                new_arch_str = self._get_string_from_ops(new_ops)
                nbhd_strs.append(new_arch_str)

        # Picks one neighbor architecture as the mutation
        mutation_str = random.choice(nbhd_strs)
        mutation_natsbenchid = self.api.archstr2index[mutation_str]
        
        return self.get([mutation_natsbenchid])

    @overrides
    def crossover(self, arch_list: List[ArchWithMetaData]) -> ArchWithMetaData:
        raise NotImplementedError
