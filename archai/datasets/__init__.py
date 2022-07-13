# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .providers.cifar10_provider import Cifar10Provider
from .providers.cifar100_provider import Cifar100Provider
from .providers.fashion_mnist_provider import FashionMnistProvider
from .providers.imagenet_provider import ImagenetProvider
from .providers.mnist_provider import MnistProvider
from .providers.svhn_provider import SvhnProvider
from .providers.food101_provider import Food101Provider
from .providers.mit67_provider import Mit67Provider
from .providers.sport8_provider import Sport8Provider
from .providers.flower102_provider import Flower102Provider
from .providers.imagenet16120_provider import ImageNet16120Provider
from .providers.synthetic_cifar10_provider import SyntheticCifar10Provider
from .providers.intel_image_provider import IntelImageProvider
from .providers.spherical_cifar100_provider import SphericalCifar100Provider
from .providers.ninapro_provider import NinaproProvider
from .providers.darcyflow_provider import DarcyflowProvider
from .providers.lmdb_image_provider import TensorpackLmdbImageProvider
from .providers.multi_lmdb_image_provider import MultiTensorpackLmdbImageProvider
from .providers.face_synthetics import FaceSyntheticsProvider
