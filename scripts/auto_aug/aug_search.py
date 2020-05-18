from archai.common.common import common_init
from archai.data_aug.search import search

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/wresnet40x2_cifar10_b512.yaml')
    search(conf)

