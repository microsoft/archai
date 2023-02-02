# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from archai.common.config import Config


def test_config():
    # Asserts that it can load keys from a YAML file
    config_file_path = "config.yaml"
    with open(config_file_path, "w") as f:
        f.write("test_key: test_value")

    config = Config(file_path=config_file_path)
    assert config["test_key"] == "test_value"

    os.remove(config_file_path)

    # Assert that additional arguments can be loaded as keys
    config = Config(
        args=["--arg1", "value1", "--arg2", "value2"],
    )
    assert config["arg1"] == "value1"
    assert config["arg2"] == "value2"
