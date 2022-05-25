from pathlib import Path
from argparse import ArgumentParser

from archai.search_spaces.discrete_search_spaces.natsbench_tss_search_spaces.discrete_search_space_natsbench_tss import DiscreteSearchSpaceNatsbenchTSS
from archai.nas.encoders.path_encoder import PathEncoder

parser = ArgumentParser('Encodes Natsbench TSS architectures using PathEncoder or path n-grams')
parser.add_argument(
    '--nats_dir', type=Path, default=Path.home() / 'dataroot/natsbench/NATS-tss-v1_0-3ffb9-simple/'
)

if __name__ == '__main__':
    args = parser.parse_args()
    
    sp = DiscreteSearchSpaceNatsbenchTSS('cifar10', str(args.nats_dir))
    model = sp.random_sample()
    print(model)

    model_graph = sp.get_arch_repr(model)
    print(model_graph)
    
    model_graph_list = [sp.get_arch_repr(sp.random_sample()) for _ in range(20)]

    pe = PathEncoder(node_features=None, path_length=-1)
    encoded_graph = pe.encode(model_graph)

    pe.fit(model_graph_list[0:10])

    print(f'Path vocab = {pe.vocab}')
    print(f'Path encoding for last 10 obs = {pe.transform(model_graph_list[10:])}')
