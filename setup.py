# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

install_requires=[
    'hyperopt', 'gorilla', 'ray>=1.0.0', 'sklearn',
    'tensorboard', 'tensorwatch>=0.9.1', 'tqdm',
    'kaleido', 'matplotlib', 'plotly', 'seaborn', 
    'h5py', 'psutil', 'pynvml', 'pyunpack', 'pyyaml', 'rarfile', 'Send2Trash',
    'overrides==3.1.0', 'runstats', 'statopt',
    'datasets', 'sacremoses', 'tokenizers>=0.10.3', 'transformers>=4.16.2',
    'onnx==1.10.2', 'onnxruntime==1.10.0',
    'coloredlogs', 'sympy', 'ftfy', # needed for text predict, fixes text for you
    'dllogger @ git+https://github.com/NVIDIA/dllogger.git',
]

setuptools.setup(
    name="archai",
    version="0.6.0",
    author="Shital Shah, Debadeepta Dey",
    author_email="shitals@microsoft.com, dedey@microsoft.com",
    description="Research platform for Neural Architecture Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/archai",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
    ],
    include_package_data=True,
    install_requires=install_requires
)
