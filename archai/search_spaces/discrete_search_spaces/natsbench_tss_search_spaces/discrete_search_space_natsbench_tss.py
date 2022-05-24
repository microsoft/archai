import random
from typing import List
from numpy import dtype
from overrides.overrides import overrides

import torch
import torch_geometric

from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import EncodableDiscreteSearchSpace
from archai.algos.natsbench.natsbench_utils import create_natsbench_tss_api, model_from_natsbench_tss
from archai.algos.natsbench.lib.models.cell_searchs import CellStructure

class DiscreteSearchSpaceNatsbenchTSS(EncodableDiscreteSearchSpace):
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

    
    def get_training_accuracy_at_n_epoch(self, 
                                        archid:int, 
                                        datasetname:str, 
                                        epoch:int)->float:
        data = self.api.query_by_index(archid, dataname=datasetname, hp='200')
        train_top1s = []
        for _, v in data.items():
            train_top1s.append(v.train_acc1es[epoch])
        
        avg_train_top1s = sum(train_top1s)/len(train_top1s)
        return avg_train_top1s


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

    
    def _get_string_from_ops(self, ops):
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # given a list of operations, get the string
        strings = ['|']
        nodes = [0, 0, 1, 0, 1, 2]
        for i, op in enumerate(ops):
            strings.append(op+'~{}|'.format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i+1] == 0:
                strings.append('+|')
        return ''.join(strings)

    @overrides
    def get_arch_repr(self, arch: ArchWithMetaData) -> torch_geometric.data.Data:
        string_rep = self.api.get_net_config(
            arch.metadata['archid'], self.datasetname
        )['arch_str']

        model_arch = list(CellStructure.str2fullstructure(string_rep).nodes)
        model_arch.insert(0, (('input', None),))
        onehot = lambda x: [int(op == x) for op in self.OPS + ['input', 'output']]

        # Node features and edges
        node_features, edges = [], []
        node_names = {}

        for out_level, out_level_nodes in enumerate(model_arch):
            node_names[out_level] = []

            for op, in_level in out_level_nodes:
                out_node = len(node_features)
                
                if in_level is not None:
                    edges += [
                        [in_node, out_node] for in_node in node_names[in_level]
                    ]

                node_names[out_level].append(out_node)
                node_features.append(onehot(op))

        # Adds output node info
        edges += [
            [in_node, len(node_features)] for in_node in node_names[out_level]
        ]
        node_features.append(onehot('output'))

        # Returns torch_geometric.data.Data object
        return torch_geometric.data.Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edges, dtype=torch.long),
            archid=arch.metadata['archid']
        )
