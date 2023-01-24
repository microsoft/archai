# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from archai.nlp.quantization.quantization_utils import rgetattr, rsetattr


@pytest.fixture
def obj():
    class DummyInnerObject:
        def __init__(self):
            self.attr = "some inner value"

    class DummyObject:
        def __init__(self):
            self.attr1 = DummyInnerObject()
            self.attr1.attr2 = DummyInnerObject()
            self.attr3 = "some value"

    return DummyObject()


def test_rgetattr(obj):
    # Assert normal attribute retrieval
    assert rgetattr(obj, "attr3") == "some value"

    # Assert recursive attribute retrieval
    assert rgetattr(obj, "attr1.attr") == "some inner value"
    assert rgetattr(obj, "attr1.attr2.attr") == "some inner value"


def test_rsetattr(obj):
    # Assert normal attribute setting
    rsetattr(obj, "attr3", "new value")
    assert obj.attr3 == "new value"

    # Assert recursive attribute setting
    rsetattr(obj, "attr1.attr", "some value")
    assert obj.attr1.attr == "some value"

    rsetattr(obj, "attr1.attr2.attr", "some value")
    assert obj.attr1.attr2.attr == "some value"
