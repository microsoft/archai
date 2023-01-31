import itertools
from pathlib import Path
from argparse import ArgumentParser

from archai.common.config import Config
from archai.discrete_search.algos import RandomSearch

from search_space.hgnet import HgnetSegmentationSearchSpace

confs_path = Path(__file__).absolute().parent / 'confs'

parser = ArgumentParser()
parser.add_argument('--search_config', type=Path, help='Search config file.', default=confs_path / 'search_config.yaml')

def filter_extra_args(extra_args: List[str], prefix: str) -> List[str]:
    return list(itertools.chain([
        [arg, val]
        for arg, val in zip(extra_args[::2], extra_args[1::2])
        if arg.startswith(prefix)
    ]))

if __name__ == '__main__':
    args, extra_args = parser.parse_known_args()

    # Filters extra args that have the prefix `search_space`
    search_extra_args = filter_extra_args(extra_args, 'search.')
    search_config = Config(str(args.search_config), search_extra_args)

    search_space = HgnetSegmentationSearchSpace(search_config)

    algo = RandomSearch(search_space, search_config)
