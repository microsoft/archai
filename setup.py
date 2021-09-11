# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

with open('README.md', 'r', encoding='utf_8') as fh:
    long_description = fh.read()

install_requires = [
    'pystopwatch2 @ git+https://github.com/ildoonet/pystopwatch2.git',
    'hyperopt',
    'tensorwatch>=0.9.1', 'tensorboard',
    'pretrainedmodels', 'tqdm', 'sklearn', 'matplotlib', 'psutil',
    'requests', 'seaborn', 'h5py', 'rarfile',
    'gorilla', 'pyyaml', 'overrides<4.0.0', 'runstats', 'psutil', 'statopt',
    'pyunpack', 'patool', 'ray>=1.0.0', 'Send2Trash'
]

setuptools.setup(
    name='archai',
    version='0.5.2',
    author='Shital Shah, Debadeepta Dey,',
    author_email='shitals@microsoft.com, dedey@microsoft.com',
    description='Research platform for Neural Architecture Search',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/archai',
    packages=setuptools.find_packages(),
        license='MIT',
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ),
    include_package_data=True,
    install_requires=install_requires
)
