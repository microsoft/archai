# Welcome to Archai

Archai is a platform for Neural Network Search (NAS) that allows you to generate efficient deep networks for your applications. Archai aspires to accelerate NAS research by enabling easy mix and match between different techniques while ensuring reproducibility, self-documented hyper-parameters and fair comparison. To achieve this, Archai uses a common code base that unifies several algorithms. Archai is extensible and modular to allow rapid experimentation of new research ideas and develop new NAS algorithms. Archai also hopes to make NAS research more accessible to non-experts by providing powerful configuration system and easy to use tools.

[Extensive feature list](docs/features.md)

## Installation

### Prerequisites

Archai requires Python 3.6+ and [PyTorch](https://pytorch.org/get-started/locally/) 1.2+. To install Python we highly recommend [Anaconda](https://www.anaconda.com/products/individual#Downloads). Archai works both on Linux as well as Windows.

### Install from source code

We recommend installing from the source code:

```bash
git clone https://github.com/microsoft/archai.git
cd archai
install.sh # on Windows, use install.bat
```

For more information, please see [Install guide](docs/install.md)

## Quick Start

### Running Algorithms

To run a specific NAS algorithm, specify it by `--algos` switch:

```bash
python scripts/main.py --algos darts --full
```

For more information on available switches and algorithms, please see [running algorithms](docs/blitz.md#running-existing-algorithms).

### Tutorials

The best way to familiarize yourself with Archai is to take a quick tour through our [30 Minute tutorial](docs/blitz.md).

We also have tbe [tutorial for Petridish](docs/petridish.md) algorithm that was developed at Microsoft Research and now available through Archai.

### Visual Studio Code

We highly recommend [Visual Studio Code](https://code.visualstudio.com/) to take advantage of predefined run configurations and interactive debugging.

From the archai directory, launch Visual Studio Code. Select the Run button (Ctrl+Shift+D), chose the run configuration you want and click on Play icon.

### Running experiments on Azure AML

To run NAS experiments at scale, you can use [Archai on Azure](tools/azure/README.md).

### Documentation

[Docs and API reference](https://microsoft.github.io/archai) is available for browsing and searching.

## Contribute

We would love community contributions, feedback, questions, algorithm implementations and feature requests! Please [file a Github issue](https://github.com/microsoft/archai/issues/new) or send us a pull request. Please review the [Microsoft Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and [learn more](https://github.com/microsoft/archai/blob/master/CONTRIBUTING.md).

## Contact

Join the Archai group on [Facebook](https://www.facebook.com/groups/1133660130366735/) to stay up to date or ask any questions.

## Team
Archai has been created and maintained by [Shital Shah](https://shitalshah.com) and [Debadeepta Dey](www.debadeepta.com) in the [Reinforcement Learning Group](https://www.microsoft.com/en-us/research/group/reinforcement-learning-redmond/) at Microsoft Research AI, Redmond, USA. Archai has benefited immensely from discussions with [John Langford](https://www.microsoft.com/en-us/research/people/jcl/), [Rich Caruana](https://www.microsoft.com/en-us/research/people/rcaruana/), [Eric Horvitz](https://www.microsoft.com/en-us/research/people/horvitz/) and [Alekh Agarwal](https://www.microsoft.com/en-us/research/people/alekha/).

We look forward to Archai becoming more community driven and including major contributors here.

## Credits

Archai builds on several open source codebases. These includes: [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment), [pt.darts](https://github.com/khanrc/pt.darts), [DARTS-PyTorch](https://github.com/dragen1860/DARTS-PyTorch), [DARTS](https://github.com/quark0/darts), [petridishnn](https://github.com/microsoft/petridishnn), [PyTorch CIFAR-10 Models](https://github.com/huyvnphan/PyTorch-CIFAR10), [NVidia DeepLearning Examples](https://github.com/NVIDIA/DeepLearningExamples), [PyTorch Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr), [NAS Evaluation is Frustratingly Hard](https://github.com/antoyang/NAS-Benchmark). Please see `install_requires` section in [setup.py](setup.py) for up to date dependencies list. If you feel credit to any material is missing, please let us know by filing a [Github issue](https://github.com/microsoft/archai/issues/new).

## License

This project is released under the MIT License. Please review the [License file](LICENSE.txt) for more details.
