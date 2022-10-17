# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import torch

class Brightness(object):
    def __init__(self, value):
        """
        Alter the Brightness of an image
        Arguments
        ---------
        value : brightness factor
            =-1 = completely black
            <0 = darker
            0 = no change
            >0 = brighter
            =1 = completely white
        """
        self.value = max(min(value,1.0),-1.0)

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = torch.clamp(_input.float().add(self.value).type(_input.type()), 0, 1)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

class RandomBrightness(object):
    def __init__(self, min_val, max_val):
        """
        Alter the Brightness of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Brightness(value)(*inputs)
        return outputs