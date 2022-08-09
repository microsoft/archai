# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf_8") as f:
    long_description = f.read()

install_requires=[
    'pystopwatch2',
    'hyperopt',
    'tensorwatch>=0.9.1', 'tensorboard',
    'pretrainedmodels', 'tqdm', 'sklearn', 'matplotlib', 'psutil',
    'requests', 'seaborn', 'h5py', 'rarfile',
    'gorilla', 'pyyaml', 'overrides==3.1.0', 'runstats', 'psutil', 'statopt',
    'pyunpack', 'patool', 'ray>=1.0.0', 'Send2Trash',
    'transformers', 'pytorch_lightning', 'tokenizers', 'datasets',
    'ftfy', # needed for text scoring, fixes text for you
    # nvidia transformer-xl
    'nv-dllogger' # same as dllogger, but PyPI installabled
    'pytorch-transformers', 'sacremoses', 'pynvml',
    'hyperopt', 'gorilla', 'ray>=1.0.0', 'sklearn',
    'tensorboard', 'tensorwatch>=0.9.1', 'tqdm',
    'kaleido', 'matplotlib', 'plotly', 'seaborn',
    'h5py', 'psutil', 'pynvml', 'pyunpack', 'pyyaml', 'rarfile', 'Send2Trash',
    'overrides==3.1.0', 'runstats', 'statopt',
    'datasets', 'sacremoses', 'tokenizers>=0.10.3', 'transformers>=4.20.1',
    'onnx==1.10.2', 'onnxruntime==1.10.0',
    'coloredlogs', 'sympy', 'ftfy', # needed for text predict, fixes text for you
]

setup(
    name="archai",
    version="0.6.5",
    description="Research platform for Neural Architecture Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/archai",
	license="MIT",
    install_requires=install_requires,
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    include_package_data=True
)
