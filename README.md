<h1 align="center">
   <img src="https://user-images.githubusercontent.com/9354770/171523113-70c7214b-8298-4d7e-abd9-81f5788f6e19.png" alt="Archai logo" width="384px" />
   <br />
</h1>

<div align="center">
   <b>Archai</b> accelerates your Neural Architecture Search (NAS) through <b>fast</b>, <b>reproducible</b> and <b>modular</b> research, allowing you to generate efficient deep networks for your applications.
</div>

<br />

<div align="center">
	<img src ="https://img.shields.io/github/release/microsoft/archai?style=flat-square" alt="Release version" />
	<img src ="https://img.shields.io/github/issues-raw/microsoft/archai?style=flat-square" alt="Open issues" />
	<img src ="https://img.shields.io/github/contributors/microsoft/archai?style=flat-square" alt="Contributors" />
	<img src ="https://img.shields.io/pypi/dm/archai?style=flat-square" alt="PyPI downloads" />
	<img src ="https://img.shields.io/github/license/microsoft/archai?color=red&style=flat-square" alt="License" />
</div>

<br />

<div align="center">
   <a href="#installation">Installation</a> •
   <a href="#quickstart">Quickstart</a> •
   <a href="#examples">Examples</a> •
   <a href="#documentation">Documentation</a> •
   <a href="#support">Support</a>
</div>

## Installation

There are various methods to install Archai, but it is recommended to use it within a virtual environment, such as `conda` or `pyenv`. This ensures that the software runs in a consistent and isolated environment, and allows for easy management of installed packages and dependencies.

PyPI provides a convenient way to install Python packages, as it allows users to easily search for and download packages, as well as automatically handle dependencies and other installation requirements. This is especially useful for larger Python projects that require multiple packages to be installed and managed.

**Archai requires Python 3.7+ and PyTorch 1.7.0+.**

```bash
pip install archai
```

Please refer to the [installation guide](https://microsoft.github.io/archai/getting_started/installation.html) for more information.

## Quickstart

To run a specific NAS algorithm, specify it by the `--algos` switch:

```terminal
python scripts/main.py --algos darts --full
```

Please refer to [available algorithms](https://microsoft.github.io/archai/advanced_guide/nas/available_algorithms.html) for more information on available switches and algorithms.

## Examples

Archai is a cutting-edge NAS platform that uses advanced Machine Learning algorithms to perform a wide range of tasks. In order to illustrate the capabilities of Archai, we will present a series of examples that showcase its ability:

* [Notebooks](https://microsoft.github.io/archai/basic_guide/notebooks.html);
* [Scripts](https://microsoft.github.io/archai/basic_guide/examples_scripts.html);
* [30-Minute Tutorial](https://microsoft.github.io/archai/basic_guide/tutorial.html);
* [Petridish](https://microsoft.github.io/archai/advanced_guide/nas/petridish.html);
* [Implementing DARTS](https://microsoft.github.io/archai/advanced_guide/nas/implementing_darts.html).

## Documentation

Please refer to the [documentation](https://microsoft.github.io/archai) for more information.

## Support

If you have any questions or feedback about the Archai project or the open problems in Neural Architecture Search, please feel free to contact us using the following information:

* Email: archai@microsoft.com
* Website: https://github.com/microsoft/archai/issues

We welcome any questions, feedback, or suggestions you may have and look forward to hearing from you.

### Team

Archai has been created and maintained by [Shital Shah](https://shital.com), [Debadeepta Dey](www.debadeepta.com), [Gustavo de Rosa](https://www.microsoft.com/en-us/research/people/gderosa), Caio Mendes, [Piero Kauffmann](https://www.microsoft.com/en-us/research/people/pkauffmann/), and [Ofer Dekel](https://www.microsoft.com/en-us/research/people/oferd) at Microsoft Research.

### Contributions

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Trademark

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

### License

This project is released under the MIT License. Please review the [file](https://github.com/microsoft/archai/blob/main/LICENSE) for more details.
