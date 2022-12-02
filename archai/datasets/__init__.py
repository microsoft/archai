# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .providers.cifar10_provider import Cifar10Provider
from .providers.cifar100_provider import Cifar100Provider
from .providers.fashion_mnist_provider import FashionMnistProvider
from .providers.imagenet_provider import ImagenetProvider
from .providers.mnist_provider import MnistProvider
from .providers.svhn_provider import SvhnProvider
from .providers.food101_provider import Food101Provider
from .providers.food101_bing_provider import Food101BingProvider
from .providers.mit67_provider import Mit67Provider
from .providers.mit67_bing_provider import Mit67BingProvider
from .providers.sport8_provider import Sport8Provider
from .providers.flower102_provider import Flower102Provider
from .providers.flower102_bing_provider import Flower102BingProvider
from .providers.aircraft_provider import AircraftProvider
from .providers.aircraft_bing_provider import AircraftBingProvider
from .providers.stanfordcars_provider import StanfordCarsProvider
from .providers.stanfordcars_bing_provider import StanfordCarsBingProvider
from .providers.person_coco_provider import PersonCocoProvider
from .providers.person_coco_cut_paste_provider import PersonCocoCutPasteProvider
from .providers.person_coco_cut_paste_clutter_provider import PersonCocoCutPasteClutterProvider