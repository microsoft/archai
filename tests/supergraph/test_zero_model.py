# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import torch

from archai.supergraph.nas.model import Model
from archai.supergraph.nas.model_desc_builder import ModelDescBuilder
from archai.common.common import common_init


def requires_gpu(test_fn: Callable):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return test_fn()
    return


@requires_gpu
def test_darts_zero_model():
    conf = common_init(config_filepath="confs/algos/darts.yaml")
    conf_search = conf["nas"]["search"]
    model_desc = conf_search["model_desc"]

    model_desc_builder = ModelDescBuilder()
    model_desc = model_desc_builder.build(model_desc)
    m = Model(model_desc, False, True)
    y, aux = m(torch.rand((1, 3, 32, 32)))
    assert isinstance(y, torch.Tensor) and y.shape == (1, 10) and aux is None


@requires_gpu
def test_petridish_zero_model():
    conf = common_init(config_filepath="confs/algos/petridish_toy.yaml")
    conf_search = conf["nas"]["search"]
    model_desc = conf_search["model_desc"]

    model_desc_builder = ModelDescBuilder()
    model_desc = model_desc_builder.build(model_desc)
    m = Model(model_desc, False, True)
    y, aux = m(torch.rand((1, 3, 32, 32)))
    assert isinstance(y, torch.Tensor) and y.shape == (1, 10) and aux is None
