# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from setuptools import find_packages, setup

install_requires = [r.rstrip() for r in open("requirements.txt", "r").readlines()]

setup(
    name="helm_harness",
    version="0.0.1",
    author="Microsoft",
    url="https://github.com/microsoft/archai/research/helm_harness",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
