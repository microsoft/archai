# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml

from archai.common.config import Config


def test_param_override1():
    conf = Config("confs/algos/darts.yaml;confs/datasets/cifar10.yaml")
    assert not conf["nas"]["eval"]["trainer"]["apex"]["enabled"]
    assert not conf["nas"]["eval"]["loader"]["apex"]["enabled"]

    conf = Config(
        "confs/algos/darts.yaml;confs/datasets/cifar10.yaml", param_args=["--nas.eval.trainer.apex.enabled", "True"]
    )
    assert conf["nas"]["eval"]["trainer"]["apex"]["enabled"]
    assert conf["nas"]["eval"]["loader"]["apex"]["enabled"]


def test_param_override2():
    conf = Config("confs/algos/darts.yaml;confs/datasets/imagenet.yaml")
    assert conf["nas"]["eval"]["trainer"]["apex"]["distributed_enabled"]
    assert conf["nas"]["eval"]["loader"]["apex"]["distributed_enabled"]

    conf = Config(
        "confs/algos/darts.yaml;confs/datasets/imagenet.yaml",
        param_args=["--nas.eval.trainer.apex.enabled", "True", "--nas.eval.trainer.apex.distributed_enabled", "False"],
    )
    assert not conf["nas"]["eval"]["trainer"]["apex"]["distributed_enabled"]
    assert not conf["nas"]["eval"]["loader"]["apex"]["distributed_enabled"]


def test_serialize():
    conf = Config()
    conf["decay"] = 1
    s = yaml.dump(conf)
    conf2 = yaml.load(s, Loader=yaml.Loader)
    assert len(conf2) == 1


def test_serialize_str():
    s = """
            !!python/object/apply:collections.OrderedDict
            - - - conf_optim
                - &id001 !!python/object:archai.common.config.Config
                  args: null
                  config_filepath: null
                  data:
                    decay: 0.0003
                    decay_bn: .nan
                    lr: 0.025
                    momentum: 0.9
                    nesterov: false
                    type: sgd
                  extra_args: []
              - - steps_per_epoch
                - 521
        """
    o = yaml.load(s, Loader=yaml.Loader)
    assert o is not None


test_serialize_str()
test_serialize()
test_param_override1()
test_param_override2()
