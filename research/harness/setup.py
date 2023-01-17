# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from setuptools import find_packages, setup

install_requires = [r.rstrip() for r in open("requirements.txt", "r").readlines()]

setup(
    name="harness",
    version="0.1",
    author="Microsoft",
    url="https://github.com/microsoft/archai/research/harness",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
