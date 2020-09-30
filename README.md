# Welcome to Archai

Archai is a platform for Neural Network Search (NAS) with a goal to unify several recent advancements in research and making them accessible to non-experts so that anyone can leverage this research to generate efficient deep networks for their own applications. Archai hopes to accelerate NAS research by easily allowing to mix and match different techniques rapidly while still ensuring reproducibility, documented hyper-parameters and fair comparison across the spectrum of these techniques. Archai is extensible and modular to accommodate new algorithms easily and aspired to offer clean and robust codebase.

[Extensive feature list](docs/features.md)

## How to Get It

### Prerequisites

Archai requires Python 3.6+ and [PyTorch](https://pytorch.org/get-started/locally/) 1.2+. To install Python we highly recommend [Anaconda](https://www.anaconda.com/products/individual#Downloads). Archai works both on Linux as well as Windows.

### Install from source code

We recommend installing from the source code:

```bash
git clone https://github.com/microsoft/archai.git
cd archai
install.sh # on Windows, use install.bat
```

For more information, please [Install guide](docs/install.md)

## How to Use It

### Quick Start

To run specific NAS algorithm, specify it by `--algos` switch:

```bash
python scripts/main.py --algos darts --full
```

For more information on available switches, algorithms etc please see [running algorithms](docs/running_algos.md).

#### Tutorial

Please see our detailed 30 minutes tutorial that walks you through how to implement Darts algorithm.

#### Visual Studio Code

We highly recommend [Visual Studio Code](https://code.visualstudio.com/) to take advantage of predefined run configurations and interactive debugging.

From archai directory, launch Visual Studio Code. Select the Run button (Ctrl+Shift+D), chose the run configuration you want and click on Play icon.

### Tutorials

### Running experiments on Azure AML

See detailed [instructions](tools/azure/README.md).

### Other References

* [Directory Structure](docs/dir_struct.md)
* [FAQ](docs/faq.md)
* [Roadmap](docs/roadmap.md)

## Contribute

We would love your contributions, feedback, questions, algorithm implementations and feature requests! Please [file a Github issue](https://github.com/microsoft/archai/issues/new) or send us a pull request. Please review the [Microsoft Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and [learn more](https://github.com/microsoft/archai/blob/master/CONTRIBUTING.md).

## Contact

Join the Archai group on [Facebook](https://www.facebook.com/groups/1133660130366735/) to stay up to date or ask any questions.

## Team
Archai has been created and maintained by [Shital Shah](https://shitalshah.com) and [Debadeepta Dey](www.debadeepta.com) in the [Reinforcement Learning Group](https://www.microsoft.com/en-us/research/group/reinforcement-learning-redmond/) at Microsoft Research AI, Redmond, USA. Archai has benefited immensely from discussions with [John Langford](https://www.microsoft.com/en-us/research/people/jcl/), [Rich Caruana](https://www.microsoft.com/en-us/research/people/rcaruana/), [Eric Horvitz](https://www.microsoft.com/en-us/research/people/horvitz/) and [Alekh Agarwal](https://www.microsoft.com/en-us/research/people/alekha/).

We look forward to Archai becoming more community driven and including major contributors here.

## Credits

Archai builds on several open source codebases. These includes: [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment), [pt.darts](https://github.com/khanrc/pt.darts), [DARTS-PyTorch](https://github.com/dragen1860/DARTS-PyTorch), [DARTS](https://github.com/quark0/darts), [petridishnn](https://github.com/microsoft/petridishnn), [PyTorch CIFAR-10 Models](https://github.com/huyvnphan/PyTorch-CIFAR10), [NVidia DeepLearning Examples](https://github.com/NVIDIA/DeepLearningExamples), [PyTorch Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr), [NAS Evaluation is Frustratingly Hard](https://github.com/antoyang/NAS-Benchmark). Please see `install_requires` section in [setup.py](setup.py) for up to date dependencies list. If you feel credit to any material is missing, please let us know by filing a [Github issue](https://github.com/microsoft/archai/issues/new).

## License

This project is released under the MIT License. Please review the [License file](LICENSE.txt) for more details.
