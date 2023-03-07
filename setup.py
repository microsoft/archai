# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re

from setuptools import find_packages, setup

dependencies = [
    "datasets>=2.4.0",
    "deepspeed",
    "einops",
    "flake8>=5.0.4",
    "gorilla>=0.4.0",
    "h5py",
    "hyperopt",
    "kaleido",
    "matplotlib",
    "nbimporter",
    "nbsphinx",
    "nbval",
    "onnx>=1.10.2",
    "onnxruntime>=1.10.0",
    "opencv-python",
    "opt_einsum",
    "overrides==3.1.0",
    "plotly",
    "psutil",
    "pytest",
    "pytorch-lightning",
    "pyyaml",
    "ray>=1.0.0",
    "scikit-learn",
    "seaborn",
    "send2trash>=1.8.0",
    "sphinx",
    "sphinx-book-theme",
    "sphinx-git",
    "sphinx-sitemap",
    "sphinx_inline_tabs",
    "sphinxcontrib-programoutput",
    "sphinxcontrib-mermaid",
    "statopt",
    "tensorboard",
    "tensorwatch",
    "tokenizers>=0.10.3",
    "torchvision",
    "transformers,"
    "tqdm",
    "transformers>=4.25.1",
]
dependencies_dict = {y: x for x, y in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in dependencies)}


def filter_dependencies(*pkgs):
    for pkg in pkgs:
        if pkg not in dependencies_dict:
            raise ValueError(f"Package {pkg} not found in dependencies")
    return [dependencies_dict[pkg] for pkg in pkgs]


extras_require = {}

extras_require["cv"] = filter_dependencies(
    "gorilla",
    "opencv-python",
    "pytorch-lightning",
    "scikit-learn",
    "torchvision",
)
extras_require["nlp"] = filter_dependencies("datasets", "einops", "opt_einsum", "tokenizers", "transformers")
extras_require["deepspeed"] = filter_dependencies("deepspeed")

extras_require["docs"] = filter_dependencies(
    "nbimporter",
    "nbsphinx",
    "sphinx",
    "sphinx-book-theme",
    "sphinx-git",
    "sphinx-sitemap",
    "sphinx_inline_tabs",
    "sphinxcontrib-programoutput",
    "sphinxcontrib-mermaid",
)
extras_require["tests"] = filter_dependencies(
    "flake8",
    "pytest",
    "einops",
    "opt_einsum",
    "git+https://github.com/sytelus/cifar_testbed.git"
)

extras_require["all"] = extras_require["cv"] + extras_require["nlp"]
if os.name != "nt":
    # Support for DeepSpeed is not available on native Windows
    extras_require["all"] += extras_require["deepspeed"]

extras_require["dev"] = extras_require["all"] + extras_require["docs"] + extras_require["tests"]

install_requires = filter_dependencies(
    "h5py",
    "hyperopt",
    "kaleido",
    "matplotlib",
    "onnx",
    "onnxruntime",
    "overrides",
    "plotly",
    "psutil",
    "pyyaml",
    "ray",
    "seaborn",
    "send2trash",
    "statopt",
    "tensorboard",
    "tensorwatch",
    "tqdm",
)

with open("README.md", "r", encoding="utf_8") as f:
    long_description = f.read()

setup(
    name="archai",
    version="1.0.0",
    description="Platform for Neural Architecture Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Microsoft",
    url="https://github.com/microsoft/archai",
    license="MIT",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
