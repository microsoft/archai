# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

from setuptools import find_packages, setup

dependencies = [
    "datasets>=2.4.0",
    "flake8>=5.0.4",
    "gorilla>=0.4.0",
    "h5py",
    "hyperopt",
    "kaleido",
    "matplotlib",
    "nbsphinx",
    "nbval",
    "onnx>=1.10.2",
    "onnxruntime>=1.10.0",
    "opencv-python",
    "overrides==3.1.0",
    "plotly",
    "psutil",
    "pytest",
    "pytorch-lightning",
    "pyunpack",
    "pyyaml",
    "ray>=1.0.0",
    "runstats>=2.0.0",
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
    "tqdm",
    "transformers>=4.25.1",
]
dependencies_dict = {y: x for x, y in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in dependencies)}


def filter_dependencies(*pkgs):
    return [dependencies_dict[pkg] for pkg in pkgs]


extras_require = {}
extras_require["cv"] = filter_dependencies(
    "gorilla",
    "opencv-python",
    "pytorch-lightning",
    "scikit-learn",
    "torchvision",
)
extras_require["nlp"] = filter_dependencies("datasets", "tokenizers", "transformers")
extras_require["all"] = extras_require["cv"] + extras_require["nlp"]

extras_require["docs"] = filter_dependencies(
    "nbsphinx",
    "sphinx",
    "sphinx-book-theme",
    "sphinx-git",
    "sphinx-sitemap",
    "sphinx_inline_tabs",
    "sphinxcontrib-programoutput",
    "sphinxcontrib-mermaid",
)
extras_require["tests"] = filter_dependencies("flake8", "nbval", "pytest")
extras_require["dev"] = extras_require["cv"] + extras_require["nlp"] + extras_require["docs"] + extras_require["tests"]

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
    "pyunpack",
    "pyyaml",
    "ray",
    "runstats",
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
