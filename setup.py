# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf_8") as f:
    long_description = f.read()

install_requires = [
    "coloredlogs>=15.0.1",
    "datasets>=2.4.0",
    "ftfy>=6.1.1",
    "gorilla>=0.4.0",
    "h5py>=3.7.0",
    "hyperopt>=0.2.7",
    "kaleido>=0.2.1",
    "matplotlib>=3.5.3",
    "nvdllogger>=1.0.0",
    "onnx==1.10.2",
    "onnxruntime==1.10.0",
    "overrides==3.1.0",
    "plotly>=5.10.0",
    "psutil>=5.9.1",
    "pynvml>=11.4.1",
    "pyunpack>=0.3",
    "pyyaml==6.0",
    "ray>=1.0.0",
    "requests==2.25.1",
    "runstats>=2.0.0",
    "sacremoses>=0.0.53",
    "scikit-learn>=1.0.2",
    "seaborn>=0.11.2",
    "send2trash>=1.8.0",
    "statopt>=0.2",
    "sympy>=1.10.1",
    "tensorboard>=2.10.0",
    "tensorwatch>=0.9.1",
    "tokenizers>=0.10.3, <=0.12.1",
    "tqdm>=4.64.0",
    "transformers>=4.16.2, <=4.20.1",
]

setup(
    name="archai",
    version="0.6.8",
    description="Platform for Neural Architecture Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/archai",
    license="MIT",
    install_requires=install_requires,
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
