# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.common import utils


class A:
    def __init__(self):
        self.a1 = 3.14


class B:
    def __init__(self):
        self.a = A()
        self.i = 3
        self.s = "eeee"
        self.d = {"k": {"kk": 5}}


def test_state_dict():
    b = B()

    sd = utils.state_dict(b)

    b.a.a1 = 0.0
    b.i = 0
    b.s = ""
    b.d = {"0": 0}

    utils.load_state_dict(b, sd)

    b0 = B()
    assert utils.deep_comp(b, b0)
