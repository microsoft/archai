# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import UserDict
import yaml
from typing import Iterator

class A(object):
    def __init__(self):
        self.hidden = 42
        self.visible = 5

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['hidden'] # cannot serialize this
        return state

a = A()
d = yaml.dump(a)
print(d)


# y = """
# a: .NaN

# """

# d=yaml.load(y, Loader=yaml.Loader)
# print(d)
# print(type( d['a']))
