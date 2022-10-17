![archai_logo_black_bg_cropped](https://user-images.githubusercontent.com/9354770/171523113-70c7214b-8298-4d7e-abd9-81f5788f6e19.png)

# Archai: Platform for Neural Architecture Search

[![License](https://img.shields.io/github/license/microsoft/archai)](https://github.com/microsoft/archai/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/microsoft/archai)](https://github.com/microsoft/archai/issues)
[![Latest release](https://img.shields.io/github/release/microsoft/archai)](https://github.com/microsoft/archai/releases)

**Archai** is a Neural Network Search (NAS) platform that allows you to generate efficient deep networks for your applications. It offers the following advantages:

* 🔬 Easy mix-and-match between different algorithms;
* 📈 Self-documented hyper-parameters and fair comparison;
* ⚡ Extensible and modular to allow rapid experimentation;
* 📂 Powerful configuration system and easy-to-use tools.

Please refer to the [documentation](https://microsoft.github.io/archai) for more information.

Package compatibility: **Python 3.7+** and **PyTorch 1.2.0+**.

OS compatibility: **Windows**, **Linux** and **MacOS**.

## Table of contents

 * [Quickstart](#quickstart)
    * [Installation](#installation)
    * [Running an Algorithm](#running-an-algorithm)
    * [Tutorials](#tutorials)
 * [Support](#support)
    * [Contributions](#contributions)
    * [Team](#team)
    * [Credits](#credits)
    * [License](#license)
    * [Trademark](#trademark)

## Quickstart

### Installation

There are many alternatives to installing Archai, but note that regardless of choice, we recommend using it within a virtual environment, such as `conda` or `pyenv`.

#### PyPI

PyPI provides a fantastic source of ready-to-go packages, and it is the easiest way to install a new package:

```terminal
pip install archai
```

#### Source (development)

Alternatively, one can clone this repository and install the bleeding-edge version:

```terminal
git clone https://github.com/microsoft/archai.git
cd archai
install.sh # on Windows, use install.bat
```

Please refer to the [installation guide](https://microsoft.github.io/archai/getting-started/install.html) for more information.

### Running an Algorithm

To run a specific NAS algorithm, specify it by the `--algos` switch:

```terminal
python scripts/main.py --algos darts --full
```

Please refer to [running algorithms](https://microsoft.github.io/archai/user-guide/tutorial.html#running-existing-algorithms) for more information on available switches and algorithms.

### Tutorials

The best way to familiarize yourself with Archai is to take a quick tour through our [30-minute tutorial](https://microsoft.github.io/archai/user-guide/tutorial.html). Additionally, one can dive into the [Petridish tutorial](https://microsoft.github.io/archai/user-guide/petridish.html) developed at Microsoft Research and available at Archai.

We highly recommend [Visual Studio Code](https://code.visualstudio.com) to take advantage of predefined run configurations and interactive debugging. From the `archai` directory, launch Visual Studio Code, select **Run** (Ctrl+Shift+D), choose the configuration and click on **Play**.

On the other hand, you can use [Archai on Azure](tools/azure/README.md) to run NAS experiments at scale.

## Support

### Contributions

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Team

Archai has been created and maintained by [Shital Shah](https://shital.com), [Debadeepta Dey](www.debadeepta.com), [Gustavo de Rosa](https://www.microsoft.com/en-us/research/people/gderosa), Caio Mendes, [Piero Kauffmann](https://www.microsoft.com/en-us/research/people/pkauffmann/), and [Ofer Dekel](https://www.microsoft.com/en-us/research/people/oferd) at Microsoft Research.

### Credits

Archai builds on several open-source codebases. These includes: [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment), [pt.darts](https://github.com/khanrc/pt.darts), [DARTS-PyTorch](https://github.com/dragen1860/DARTS-PyTorch), [DARTS](https://github.com/quark0/darts), [petridishnn](https://github.com/microsoft/petridishnn), [PyTorch CIFAR-10 Models](https://github.com/huyvnphan/PyTorch-CIFAR10), [NVidia DeepLearning Examples](https://github.com/NVIDIA/DeepLearningExamples), [PyTorch Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr), [NAS Evaluation is Frustratingly Hard](https://github.com/antoyang/NAS-Benchmark), [NASBench-PyTorch](https://github.com/romulus0914/NASBench-PyTorch).

Please see `install_requires` section in [setup.py](https://github.com/microsoft/archai/blob/master/setup.py) for up-to-date dependencies list. If you feel credit to any material is missing, please let us know by filing an [issue](https://github.com/microsoft/archai/issues).

### License

This project is released under the MIT License. Please review the [file](https://github.com/microsoft/archai/blob/master/LICENSE) for more details.

### Trademark

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
