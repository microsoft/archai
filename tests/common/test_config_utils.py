# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml

from archai.common import config_utils
from archai.supergraph.utils import utils


def test_yaml1():
    input = """
    l1:
        l1c1: 5
        l2:
            l2c2: 6
            l3:
                l3c2: 7
                l3c3: 8

    n1:
        _copy: '/l1'

    n2:
        n22:
            _copy: '/l1/l2/l3'

    n3: '_copy: /n1/l1c1'

    n4:
        n4c1: 9
        n4c2: '_copy: ../n4c1'
        n4c3: '_copy: ../../n2/n22'


    n5: '_copy: ./../n1/l1c1'
    """

    expected = """
    l1:
        l1c1: 5
        l2: &id001
            l2c2: 6
            l3:
                l3c2: 7
                l3c3: 8
    n1:
        l1c1: 5
        l2: *id001
    n2:
        n22: &id002
            l3c2: 7
            l3c3: 8
    n3: 5
    n4:
        n4c1: 9
        n4c2: 9
        n4c3: *id002
    n5: 5
    """

    d_input = yaml.safe_load(input)
    config_utils.resolve_all(d_input)
    # print(yaml.dump(d))
    d_expected = yaml.safe_load(expected)

    utils.deep_comp(d_input, d_expected)


def test_yaml2():
    input = """
    common:
        dataset:
            name: 'd1'
            classes: 3
            cifar:
                limit: -1
        logging: True

    autoaug:
        dataset: '_copy: /nas/train/dataset'
        logging: '_copy: /nas/logging'

    nas:
        train:
            dataset:
                _copy: '/common/dataset'
                classes: 4
                cifar:
                    workers: 0
        logging: '_copy: /common/logging'
    """

    expected = """
        autoaug:
            dataset: &id001
                cifar:
                    limit: -1
                    workers: 0
                classes: 4
                name: d1
            logging: true
        common:
            dataset:
                cifar:
                    limit: -1
                classes: 3
                name: d1
            logging: true
        nas:
            logging: true
            train:
                dataset: *id001
    """

    d_input = yaml.safe_load(input)
    # print(d)
    config_utils.resolve_all(d_input)
    d_expected = yaml.safe_load(expected)

    assert utils.deep_comp(d_input, d_expected)


test_yaml1()
test_yaml2()
