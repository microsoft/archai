from archai.common.common import common_init
from archai.datasets.providers.cifar10_provider import Cifar10Provider
from archai.datasets.meta_dataset import MetaDataset

def main():
    conf = common_init(config_filepath='confs/algos/resnet.yaml')

    provider = Cifar10Provider(conf['dataset'])
    train_transform, test_transform = provider.get_transforms()
    trainset, testset = provider.get_datasets(load_train=True, load_test=True,
                     transform_train=None, transform_test=None)

    trainset = MetaDataset(trainset, transform=train_transform)

    for x,y,meta in trainset:
        print(x,y,meta)

if __name__ == '__main__':
    main()