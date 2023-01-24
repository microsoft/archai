from archai.supergraph.utils.common import common_init
from archai.supergraph.utils.augmented_searcher import search

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/aug/wresnet40x2_cifar10_b512.yaml')
    search(conf)

