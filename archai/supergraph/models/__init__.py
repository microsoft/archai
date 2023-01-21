import torch

from torch import nn
from torch.nn import DataParallel
# from torchvision import models

from archai.supergraph.nas.models.resnet import ResNet
from archai.supergraph.nas.models.pyramidnet import PyramidNet
from archai.supergraph.nas.models.shakeshake.shake_resnet import ShakeResNet
from archai.supergraph.nas.models.wideresnet import WideResNet
from archai.supergraph.nas.models.shakeshake.shake_resnext import ShakeResNeXt

from archai.supergraph.nas.models.mobilenetv2 import *
from archai.supergraph.nas.models.resnet_cifar10 import *
from archai.supergraph.nas.models.vgg import *
from archai.supergraph.nas.models.densenet import *
from archai.supergraph.nas.models.resnet_orig import *
from archai.supergraph.nas.models.googlenet import *
from archai.supergraph.nas.models.inception import *


def get_model(conf, num_class=10):
    name = conf['type']

    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, n_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, n_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, n_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, n_classes=num_class)

    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_class)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_class)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)

    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)

    elif name == 'pyramid':
        model = PyramidNet('cifar10', depth=conf['depth'], alpha=conf['alpha'], n_classes=num_class, bottleneck=conf['bottleneck'])
    else:
        raise NameError('no model named, %s' % name)


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]