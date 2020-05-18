# Welcome to Archai

Neural Architecture Search (NAS) aims to automate the process of searching for neural architectures.
Given a new dataset, it is often a tedious task of trying out many different architectures and
hyperparameters manually. Even the most skilled machine learning researchers and engineers have to
resort to the dark arts of finding good architectures and corresponding hyperparameters guided by some
intuition and a lot of careful experimentation. The NAS community's dream is that this tedium be taken
over by algorithms, freeing up precious human time for more noble pursuits.

Recently, NAS has made tremendous progress but is merely getting started. Many open problems remain.
But one of the more immediate problems is fair comparison and reproducibility. To ameliorate these
issues we are releasing Archai which is a performant platform for NAS algorithms. Archai has the following features:

Arhai has the following features:

* NAS for non-experts
    * Turnkey experimentation platform
* High performance PyTorch code base
* Ease of algorithm development
    * Object-oriented model definition
    * Unified abstractions for training and evaluation
    * New algorithms can be written in a few lines of code
    * Easily mix and match existing algorithm aspects
    * Easily implement both forward and backward search
    * Algorithm-agnostic pareto-front generation
        * Easily add hardware-specific constraints like memory, inference time, flops etc.

* Efficient experiment management for reproducibility and fair comparison
    * Flexible configuration system
    * Structured logging
    * Metrics management and logging
    * Declarative experimentation
    * Declarative support for wide variety of datasets
    * Custom dataset support
    * Unified final training procedure for searched models

## Installation

Currently we have tested Archai on Ubuntu 16.04 LTS 64-bit and Ubuntu 18.04 LTS 64-bit on
Python 3.6+ and PyTorch 1.3+.

* System prep:
    * CUDA compatible GPU
    * Anaconda package manager
    * Nvidia driver compatible with cuda 9.2 or greater
* We provide two conda environments for cuda 9.2 and cuda 10.1.
    * [archaicuda101.yaml](dockers/docker-cuda-10-1/archaicuda101.yml)
    * [archaicuda92.yaml](dockers/docker-cuda-9-2/archaicuda92.yml)

* `conda env create -f archaicuda101.yml`
* `conda activate archaicuda101`
* `pip install -r requirements.txt`
* `pip install -e .`

## Test installation

* `cd archai`
* The below command will run every algorithm through a few batches of cifar10
  and for both search and final training
* `python scripts/main.py`
* If all went well, now you have a working installation!
* Note one can also build and use the cuda 10.1 or 9.2 compatible dockers
  provided in the [dockers](dockers) folder. These dockers are useful
  for large scale experimentation on compute clusters.


## How to use it?

```
├── archai
│   ├── cifar10_models
│   ├── common
│   ├── darts
│   ├── data_aug
│   ├── nas
│   ├── networks
│   ├── petridish
│   ├── random
│   └── xnas
├── archived
├── confs
├── dockers
├── docs
├── scripts
├── setup.py
├── tests
└── tools
    ├── azure
```

Most of the functionality resides in the [`archai`](archai/) folder.
[`nas`](archai/nas) contains algorithm-agnostic infrastructure
that is commonly used in NAS algorithms. [`common`](archai/common) contains
common infrastructure code that has no nas-specific code but is infrastructre
code that gets widely used.
Algorithm-specific code resides in appropriately named folder like [`darts`](archai/nas/darts),
[`petridish`](archai/nas/petridish), [`random`](archai/nas/random),
[`xnas`](archai/nas/xnas)

[`scripts`](archai/scripts) contains entry-point scripts to running all algorithms.

### Quick start

[`scripts/main.py`](archai/scripts/main.py) is the main point of entry.

#### Run all algorithms in toy mode

`python scripts/main.py` runs all implemented search algorithms and final training
with a few minibatches of data from cifar10. This is designed to exercise all
code paths and make sure that everything is properly.

#### To run specific algorithms

`python scripts/main.py --darts` will run darts search and evaluation (final model training) using only a few minibatches of data from cifar10.
`python scripts/main.py --darts --full` will run the full search.

Other algorithms can be run by specifying different algorithm names like `petridish`, `xnas`, `random` etc.

#### List of algorithms

Current the following algorithms are implemented:

* [Petridish](https://papers.nips.cc/paper/9202-efficient-forward-architecture-search.pdf)
* [DARTS](https://deepmind.com/research/publications/darts-differentiable-architecture-search)
* [Random search baseline]
* [XNAS](http://papers.nips.cc/paper/8472-xnas-neural-architecture-search-with-expert-advice.pdf) (this is currently experimental and has not been fully reproduced yet as XNAS authors have not released source code at the time of writing.)

See [Roadmap](#roadmap) for details on new algorithms coming soon.

### Tutorials

### Running experiments on Azure AML

See detailed [instructions](tools/azure/README.md).

## Roadmap

We are striving to rapidly update the list of algorithms and encourage pull-requests from the community
of new algorithms.

Here is our current deck:

* [ProxyLess NAS](https://arxiv.org/abs/1812.00332)
* [SNAS](https://arxiv.org/abs/1812.09926)
* [DATA](http://papers.nips.cc/paper/8374-data-differentiable-architecture-approximation.pdf)
* [RandNAS](https://liamcli.com/assets/pdf/randnas_arxiv.pdf)

Please file in the issues algorithms you would like to see implemented in Archai. We will try our best to accomodate.

## Paper
If you use Archai in your work please cite...

## Contribute

We would love your contributions, feedback, questions, and feature requests! Please file a github issue or send us a pull request. Please review the [Microsoft Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and [learn more](CONTRIBUTING.md).

## Contacts

Shital Shah shitals@microsoft.com

Debadeepta Dey dedey@microsoft.com

Eric Horvitz horvitz@microsoft.com

## Credits

Archai utilizes several open source libraries for many of its features. These includes:[fastautoaugment](https://github.com/kakaobrain/fast-autoaugment), [tensorwatch](https://github.com/microsoft/tensorwatch), and many others.

## License
This project is released under the MIT License. Please review the [License file](LICENSE.txt) for further details.


