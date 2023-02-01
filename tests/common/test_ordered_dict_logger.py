# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from archai.common.ordered_dict_logger import OrderedDictLogger


def test_ordered_dict_logger():
    # Assert that the default attributes are defined
    logger = OrderedDictLogger(file_path="log.yaml", delay=0.0)
    assert logger.file_path == "log.yaml"
    assert logger.delay == 0.0
    assert isinstance(logger.root_node, dict)
    assert len(logger.root_node) == 0
    assert logger.current_path == ""

    # Assert that the updated key is defined
    logger._update_key("test_key", "test_value")
    assert len(logger.root_node) == 1
    assert logger.root_node["test_key"] == "test_value"
    assert logger.current_path == ""

    # Assert that the logger can be saved
    logger.save()
    assert os.path.exists("log.yaml")

    # Assert that the logger can be loaded
    logger = OrderedDictLogger(delay=0.0)
    logger.load("log.yaml")
    assert len(logger.root_node) == 1
    assert logger.root_node["test_key"] == "test_value"
    if os.path.exists("log.yaml"):
        os.remove("log.yaml")
