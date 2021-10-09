# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools, platform

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

install_requires=[
    'pystopwatch2 @ git+https://github.com/ildoonet/pystopwatch2.git',
    'hyperopt', #  @ git+https://github.com/hyperopt/hyperopt.git
    'tensorwatch>=0.9.1', 'tensorboard',
    'pretrainedmodels', 'tqdm', 'sklearn', 'matplotlib', 'psutil',
    'requests', 'seaborn', 'h5py', 'rarfile',
    'gorilla', 'pyyaml', 'overrides', 'runstats', 'psutil', 'statopt',
    'pyunpack', 'patool', 'ray>=1.0.0', 'Send2Trash',
    'transformers', 'pytorch_lightning', 'tokenizers', 'datasets',
    'ftfy', # needed for text scoring, fixes text for you
    # nvidia transformer-xl
    'dllogger @ git+https://github.com/NVIDIA/dllogger.git',
    'pytorch-transformers', 'sacremoses', 'pynvml'
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
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
